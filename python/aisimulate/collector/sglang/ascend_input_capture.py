"""Observe source-selected operator geometry alongside Ascend routing capture.

SGLang baa9befa381511cd4d56f83980951e5a45fffc6d uses static/dynamic
linear_method_npu.apply, deepseek_v2_attention_mla_npu.forward_dsa_prepare/core
and the NPU lightning-indexer/sparse-attention entry points below. Only small
sequence-length tensors remain referenced for replay snapshots. These hooks
produce invocation metadata, never performance measurements.
"""

import dataclasses
import functools
import importlib
import json
import os
from enum import Enum


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
            elif label == "cache_locations" and value.numel() <= 256:
                result["captured_values"] = value
            return result
        if isinstance(value, (list, tuple)):
            return [self.describe(item, label) for item in value]
        if isinstance(value, self.torch.nn.Module):
            return {
                "python_type": type(value).__module__ + "." + type(value).__name__,
                "parameters": {
                    name: self.describe(parameter, name) for name, parameter in value.named_parameters(recurse=False)
                },
                "attributes": {
                    name: self.describe(getattr(value, name), name)
                    for name in ("variance_epsilon", "eps", "normalized_shape")
                    if hasattr(value, name)
                },
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "python_type": type(value).__module__ + "." + type(value).__name__,
                "fields": {
                    field.name: self.describe(getattr(value, field.name), field.name)
                    for field in dataclasses.fields(value)
                },
            }
        if isinstance(value, dict):
            return {str(key): self.describe(item, str(key)) for key, item in value.items()}
        if isinstance(value, Enum):
            return {"python_type": type(value).__module__ + "." + type(value).__name__, "name": value.name}
        if isinstance(value, (set, frozenset)):
            return [self.describe(item, label) for item in sorted(value, key=str)]
        if isinstance(value, functools.partial):
            return {
                "function": self.describe(value.func),
                "arguments": self.describe(value.args),
                "keywords": self.describe(value.keywords),
            }
        if hasattr(value, "__struct_fields__"):
            return {
                "python_type": type(value).__module__ + "." + type(value).__name__,
                "fields": {name: self.describe(getattr(value, name), name) for name in value.__struct_fields__},
            }
        if callable(value):
            return {"callable": value.__module__ + "." + getattr(value, "__qualname__", type(value).__name__)}
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
            arguments = {"positional": args, **kwargs}
            if kind == "torch_npu.npu_scatter_nd_update_":
                if len(args) < 3:
                    raise ValueError("Expected the audited positional cache scatter signature")
                arguments["cache_locations"] = args[1]
            observer.record(kind, arguments, output)
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
    wrap_function(
        torch.ops.npu,
        "batch_matmul_transpose",
        "torch.ops.npu.batch_matmul_transpose",
        preserve_attributes=True,
    )

    if os.environ.get("AISIM_CAPTURE_VECTORS") == "1":
        for name in ("matmul", "cat"):
            wrap_function(torch, name, "torch." + name)
        for name in ("layer_norm", "embedding"):
            wrap_function(torch.nn.functional, name, "torch.nn.functional." + name)
        for name in (
            "npu_rms_norm",
            "npu_add_rms_norm",
            "npu_swiglu",
            "npu_scatter_nd_update_",
            "npu_rotary_mul",
            "npu_mrope",
        ):
            wrap_function(torch_npu, name, "torch_npu." + name)
        wrap_function(torch.ops.npu, "npu_moe_gating_top_k", "torch.ops.npu.npu_moe_gating_top_k", True)
        for module_name, function_name in (
            ("sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu", "fused_split_qk_norm"),
            ("sglang.srt.layers.rotary_embedding.base", "fused_rope_qk_mqa"),
            ("sglang.srt.hardware_backend.npu.moe.topk", "fused_topk_npu"),
        ):
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
            if function is None:
                raise ValueError(f"Expected pinned NPU vector implementation: {module_name}.{function_name}")
            wrap_function(module, function_name, function.__module__ + "." + function.__name__)
        activation = importlib.import_module("sglang.srt.hardware_backend.npu.moe.activation")
        wrap_function(activation.NPUSwigluDeepEPKernel, "_apply_activation", "NPUSwigluDeepEPKernel._apply_activation")
        dispatcher = importlib.import_module("sglang.srt.layers.moe.token_dispatcher.deepep")
        for name in ("dispatch_a", "dispatch_b", "combine_a", "combine_b"):
            wrap_function(dispatcher.DeepEPDispatcher, name, "DeepEPDispatcher." + name)

    if os.environ.get("AISIM_CAPTURE_BOUNDARIES") == "1":
        from sglang.srt.distributed.parallel_state import GroupCoordinator
        from sglang.srt.layers.layer_boundary.stage import StageBoundary

        prepare = StageBoundary.prepare

        @functools.wraps(prepare)
        def prepare_stage(boundary, hidden_states, forward_batch, **kwargs):
            output = prepare(boundary, hidden_states, forward_batch, **kwargs)
            observer.record(
                "StageBoundary.prepare",
                {
                    "kind": boundary.kind,
                    "path": boundary.plan.path_for(forward_batch),
                    "input": hidden_states,
                },
                output,
                boundary.norm,
            )
            return output

        StageBoundary.prepare = prepare_stage
        hooks.append("StageBoundary.prepare")

        def wrap_collective(name):
            original = getattr(GroupCoordinator, name)

            @functools.wraps(original)
            def collective(group, *args, **kwargs):
                depth = getattr(state.thread, "collective_depth", 0)
                state.thread.collective_depth = depth + 1
                try:
                    output = original(group, *args, **kwargs)
                    if depth == 0:
                        observer.record(
                            "GroupCoordinator." + name,
                            {"ranks": group.ranks, "rank_in_group": group.rank_in_group, "positional": args, **kwargs},
                            output,
                        )
                    return output
                finally:
                    state.thread.collective_depth = depth

            setattr(GroupCoordinator, name, collective)
            hooks.append("GroupCoordinator." + name)

        for name in ("all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter", "reduce_scatter_tensor"):
            wrap_collective(name)

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
