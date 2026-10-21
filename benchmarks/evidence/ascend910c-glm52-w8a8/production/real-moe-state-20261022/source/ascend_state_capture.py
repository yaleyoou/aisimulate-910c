"""Observe actual Ascend MoE input counts, including captured graph replays.

This observer adds count copies and synchronization; its runs supply invocation
state and numerical evidence, never uninstrumented performance measurements.
Dispatch contract: SGLang baa9befa381511cd4d56f83980951e5a45fffc6d,
srt/layers/moe/moe_runner/ascend.py:319/349 and
srt/hardware_backend/npu/quantization/moe_methods.py:681.
"""

import functools
import importlib
import json
import os
import re
import threading
from pathlib import Path


class CaptureState:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.thread = threading.local()
        self.graphs = {}
        self.eager = {}
        self.case_id = None
        self.recorded = 0
        self.weight_names = {}

    def control(self):
        path = self.directory / "control.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text())
        case = value.get("case_id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case):
            raise ValueError("Capture case_id must be a simple filename component")
        if self.case_id != case:
            self.case_id = case
            self.recorded = 0
        return value if value.get("enabled") and self.recorded < int(value["max_forwards"]) else None

    def capture(self, quant_info, hidden_states, expert_tokens, output_dtype, group_list_type):
        graph = getattr(self.thread, "capturing_graph", None)
        if graph is None and not getattr(self.thread, "recording_forward", False):
            return
        weight = quant_info.w13_weight
        entry = {
            "counts": expert_tokens.clone(),
            "input_shape": list(hidden_states.shape),
            "input_dtype": str(hidden_states.dtype),
            "weight_shape": list(weight.shape),
            "weight_dtype": str(weight.dtype),
            "output_dtype": str(output_dtype),
            "group_list_type": group_list_type,
        }
        destination = self.graphs.setdefault(graph, {}) if graph is not None else self.eager
        if weight.data_ptr() in destination:
            raise ValueError("Repeated expert projection in one capture requires an explicit invocation dimension")
        destination[weight.data_ptr()] = entry

    def write(self, runner, forward_batch, control, torch):
        if not self.weight_names:
            self.weight_names = {
                parameter.data_ptr(): name
                for name, parameter in runner.model.named_parameters()
                if name.endswith("w13_weight")
            }
        graph = getattr(self.thread, "replayed_graph", None)
        captured = self.graphs.get(graph, {}) if graph is not None else self.eager
        if not captured:
            raise ValueError("No captured MoE inputs for this forward")
        torch.npu.synchronize()
        rows = []
        for pointer, entry in captured.items():
            name = self.weight_names.get(pointer)
            if name is None:
                raise ValueError("Captured weight cannot be mapped to the loaded model")
            values = entry["counts"].cpu().tolist()
            if entry["group_list_type"] != 1 or any(value < 0 for value in values):
                raise ValueError("Expected nonnegative per-expert counts from the Ascend runner")
            if len(values) != entry["weight_shape"][0] or sum(values) > entry["input_shape"][0]:
                raise ValueError("Expert counts do not fit the actual grouped-matmul storage")
            rows.append({"parameter": name, **{key: value for key, value in entry.items() if key != "counts"},
                         "expert_counts": values, "valid_rows": sum(values)})
        expected = len(self.weight_names)
        if len(rows) != expected:
            raise ValueError(f"Incomplete MoE layer capture: {len(rows)} of {expected}")
        rank = torch.distributed.get_rank()
        output = self.directory / control["case_id"]
        output.mkdir(parents=True, exist_ok=True)
        lengths = forward_batch.seq_lens
        record = {
            "schema": "aisimulate-ascend-moe-state/v1", "scope": "state_only_not_performance",
            "rank": rank, "ordinal": self.recorded, "forward_mode": forward_batch.forward_mode.name,
            "batch_size": forward_batch.batch_size,
            "input_tokens": forward_batch.input_ids.numel() if forward_batch.input_ids is not None else 0,
            "seq_lens": lengths.cpu().tolist() if lengths is not None else [],
            "global_num_tokens_cpu": forward_batch.global_num_tokens_cpu,
            "execution": "graph" if graph is not None else "eager", "layers": rows,
        }
        with (output / f"rank-{rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.recorded += 1


def install(directory=None):
    directory = directory or os.environ.get("AISIM_CAPTURE_DIR")
    if not directory:
        raise ValueError("AISIM_CAPTURE_DIR is required for the recording launcher")
    import torch
    import torch_npu

    importlib.import_module("sglang.srt.layers.quantization")
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
    from sglang.srt.model_executor.model_runner import ModelRunner

    state = CaptureState(directory)
    graph_class = torch_npu.npu.NPUGraph
    begin, end, replay = graph_class.capture_begin, graph_class.capture_end, graph_class.replay

    @functools.wraps(begin)
    def capture_begin(graph, *args, **kwargs):
        result = begin(graph, *args, **kwargs)
        state.thread.capturing_graph = id(graph)
        state.graphs[id(graph)] = {}
        return result

    @functools.wraps(end)
    def capture_end(graph, *args, **kwargs):
        try:
            return end(graph, *args, **kwargs)
        finally:
            state.thread.capturing_graph = None

    @functools.wraps(replay)
    def graph_replay(graph, *args, **kwargs):
        state.thread.replayed_graph = id(graph)
        return replay(graph, *args, **kwargs)

    graph_class.capture_begin, graph_class.capture_end, graph_class.replay = capture_begin, capture_end, graph_replay
    original_apply = NPUW8A8Int8MoEMethod.apply

    @functools.wraps(original_apply)
    def apply(kernel, quant_info, hidden_states, expert_tokens, pertoken_scale, output_dtype, weight_prefix, group_list_type):
        result = original_apply(kernel, quant_info, hidden_states, expert_tokens, pertoken_scale,
                                output_dtype, weight_prefix, group_list_type)
        if weight_prefix == "w13":
            state.capture(quant_info, hidden_states, expert_tokens, output_dtype, group_list_type)
        return result

    NPUW8A8Int8MoEMethod.apply = apply
    original_forward = ModelRunner.forward

    @functools.wraps(original_forward)
    def forward(runner, forward_batch, *args, **kwargs):
        control = state.control()
        state.thread.recording_forward = control is not None
        state.thread.replayed_graph = None
        state.eager = {}
        try:
            result = original_forward(runner, forward_batch, *args, **kwargs)
            if control is not None:
                state.write(runner, forward_batch, control, torch)
            return result
        except Exception as error:
            if control is not None:
                state.directory.mkdir(parents=True, exist_ok=True)
                (state.directory / f"error-{os.getpid()}.json").write_text(json.dumps({
                    "case_id": control["case_id"], "error": f"{type(error).__name__}: {error}"
                }))
            raise
        finally:
            state.thread.recording_forward = False

    ModelRunner.forward = forward
    state.directory.mkdir(parents=True, exist_ok=True)
    (state.directory / f"installed-{os.getpid()}.json").write_text(json.dumps({
        "pid": os.getpid(), "scope": "state_only_not_performance",
        "hooks": ["NPUW8A8Int8MoEMethod.apply", "ModelRunner.forward", "NPUGraph.capture_begin", "NPUGraph.replay"],
    }) + "\n")
    return state
