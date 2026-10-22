"""Observe source-selected operator geometry alongside Ascend routing capture.

SGLang baa9befa381511cd4d56f83980951e5a45fffc6d uses static/dynamic
linear_method_npu.apply, deepseek_v2_attention_mla_npu.forward_dsa_prepare/core
and the NPU lightning-indexer/sparse-attention entry points below. Only small
sequence-length tensors remain referenced for replay snapshots. These hooks
produce invocation metadata, never performance measurements.
"""

import functools
import json
import os


class ObservedCallable:
    def __init__(self, original, invoke):
        self.original = original
        self.invoke = invoke

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.original, name)


class InputCapture:
    def __init__(self, state, torch, torch_npu):
        self.state = state
        self.torch = torch
        self.torch_npu = torch_npu
        self.graphs = {}
        self.eager = []
        self.parameter_names = {}
        self.module_names = {}

    def describe(self, value, label=""):
        if isinstance(value, self.torch.Tensor):
            result = {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "data_ptr": value.data_ptr(),
                "format": self.torch_npu.get_npu_format(value) if value.device.type == "npu" else None,
            }
            if "seq_lengths" in label:
                if value.numel() > 256:
                    raise ValueError("Unexpected large sequence-length metadata")
                result["captured_values"] = value
            return result
        if isinstance(value, (list, tuple)):
            return [self.describe(item, label) for item in value]
        if isinstance(value, dict):
            return {str(key): self.describe(item, str(key)) for key, item in value.items()}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return {"python_type": type(value).__module__ + "." + type(value).__name__}

    def record(self, kind, arguments, output, module=None):
        graph = getattr(self.state.thread, "capturing_graph", None)
        if graph is None and not getattr(self.state.thread, "recording_forward", False):
            return
        destination = self.graphs.setdefault(graph, []) if graph is not None else self.eager
        destination.append(
            {
                "kernel": kind,
                "module_id": id(module) if module is not None else None,
                "arguments": self.describe(arguments),
                "output": self.describe(output),
            }
        )

    def snapshot(self, graph=None):
        values_cache = {}

        def render(value):
            if isinstance(value, self.torch.Tensor):
                key = (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype))
                if key not in values_cache:
                    values_cache[key] = value.detach().cpu().tolist()
                return values_cache[key]
            if isinstance(value, list):
                return [render(item) for item in value]
            if isinstance(value, dict):
                result = {key: render(item) for key, item in value.items()}
                if "data_ptr" in result and result["data_ptr"] in self.parameter_names:
                    result["parameter"] = self.parameter_names[result["data_ptr"]]
                if result.get("module_id") in self.module_names:
                    result["module"] = self.module_names[result["module_id"]]
                return result
            return value

        return render(self.graphs.get(graph, []) if graph is not None else self.eager)


def install(state):
    import torch
    import torch_npu
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUW8A8Int8DynamicLinearMethod,
        NPUW8A8Int8LinearMethod,
    )
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.model_executor.model_runner import ModelRunner

    observer = InputCapture(state, torch, torch_npu)
    hooks = []

    def wrap_function(owner, name, kind, preserve_attributes=False):
        original = getattr(owner, name)

        @functools.wraps(original)
        def invoke(*args, **kwargs):
            output = original(*args, **kwargs)
            observer.record(kind, {"positional": args, **kwargs}, output)
            return output

        setattr(owner, name, ObservedCallable(original, invoke) if preserve_attributes else invoke)
        hooks.append(kind)

    def wrap_linear(kernel_type):
        original = kernel_type.apply
        kind = kernel_type.__module__ + "." + kernel_type.__name__

        @functools.wraps(original)
        def invoke(kernel, layer, inputs, bias=None):
            output = original(kernel, layer, inputs, bias)
            group = getattr(layer, "tp_group", None)
            has_quant_bias = hasattr(layer, "quant_bias") and not (
                isinstance(layer, RowParallelLinear) and group is not None and group.rank_in_group > 0
            )
            observer.record(
                kind,
                {"input": inputs, "weight": layer.weight, "quant_bias": has_quant_bias, "bias": bias},
                output,
                layer,
            )
            return output

        kernel_type.apply = invoke
        hooks.append(kind)

    for kernel_type in (NPUW8A8Int8LinearMethod, NPUW8A8Int8DynamicLinearMethod):
        wrap_linear(kernel_type)
    wrap_function(torch.nn.functional, "linear", "torch.nn.functional.linear")
    for name in ("npu_transpose_batchmatmul", "npu_lightning_indexer", "npu_sparse_flash_attention"):
        wrap_function(torch_npu, name, "torch_npu." + name)
    wrap_function(torch.ops.npu, "batch_matmul_transpose", "torch.ops.npu.batch_matmul_transpose", preserve_attributes=True)

    original_begin = torch_npu.npu.NPUGraph.capture_begin

    @functools.wraps(original_begin)
    def begin(graph, *args, **kwargs):
        result = original_begin(graph, *args, **kwargs)
        observer.graphs[id(graph)] = []
        return result

    torch_npu.npu.NPUGraph.capture_begin = begin
    original_forward = ModelRunner.forward

    @functools.wraps(original_forward)
    def forward(runner, *args, **kwargs):
        observer.eager = []
        return original_forward(runner, *args, **kwargs)

    ModelRunner.forward = forward
    original_write = state.write

    def write(runner, forward_batch, control, torch_module):
        ordinal = state.recorded
        original_write(runner, forward_batch, control, torch_module)
        if not observer.parameter_names:
            observer.parameter_names = {value.data_ptr(): name for name, value in runner.model.named_parameters()}
            observer.module_names = {id(module): name for name, module in runner.model.named_modules()}
        graph = getattr(state.thread, "replayed_graph", None)
        calls = observer.snapshot(graph)
        if not calls:
            raise ValueError("No operator inputs observed for selected forward")
        rank = torch.distributed.get_rank()
        record = {
            "schema": "aisimulate-ascend-operator-inputs/v1",
            "scope": "state_only_not_performance",
            "rank": rank,
            "ordinal": ordinal,
            "execution": "graph" if graph is not None else "eager",
            "forward_mode": forward_batch.forward_mode.name,
            "batch_size": forward_batch.batch_size,
            "global_num_tokens_cpu": forward_batch.global_num_tokens_cpu,
            "calls": calls,
        }
        with (state.directory / control["case_id"] / f"inputs-rank-{rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    state.write = write
    (state.directory / f"inputs-installed-{os.getpid()}.json").write_text(
        json.dumps({"scope": "state_only_not_performance", "hooks": hooks}) + "\n"
    )
    return observer
