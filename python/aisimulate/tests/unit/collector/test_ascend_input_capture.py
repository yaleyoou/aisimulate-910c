import dataclasses
import json
import weakref
from types import SimpleNamespace

import pytest
import torch

from collector.sglang.ascend_input_capture import InputCapture, ObservedCallable

pytestmark = pytest.mark.unit


def observer(graph=None, eager=True):
    state = SimpleNamespace(thread=SimpleNamespace(capturing_graph=graph, recording_forward=eager))
    return InputCapture(state, torch, None)


def test_graph_snapshot_observes_current_lengths_without_retaining_large_activations():
    capture = observer(graph=7)
    activations = torch.ones((8, 6144))
    reference = weakref.ref(activations)
    lengths = torch.tensor([8192, 1, 1, 1, 1, 1, 1, 1])
    capture.record("indexer", {"query": activations, "actual_seq_lengths_key": lengths}, None)
    del activations
    assert reference() is None
    lengths[0] = 9000
    result = capture.snapshot(7)
    assert result[0]["arguments"]["actual_seq_lengths_key"]["captured_values"] == [9000, 1, 1, 1, 1, 1, 1, 1]
    assert result[0]["arguments"]["query"]["shape"] == [8, 6144]
    json.dumps(result)


def test_inactive_forward_is_ignored_and_unexpected_large_lengths_fail():
    capture = observer(eager=False)
    capture.record("linear", {}, None)
    assert capture.snapshot() == []
    capture.state.thread.recording_forward = True
    with pytest.raises(ValueError, match="large sequence"):
        capture.record("indexer", {"actual_seq_lengths_query": torch.ones(257)}, None)


def test_named_parameter_views_and_multiple_invocations_are_preserved():
    capture = observer()
    weight = torch.ones((4, 8))
    module = SimpleNamespace()
    capture.parameter_names[weight.data_ptr()] = "model.router.weight"
    capture.module_names[id(module)] = "model.router"
    for _ in range(2):
        capture.record("linear", {"weight": weight.transpose(0, 1)}, None, module)
    rows = capture.snapshot()
    assert len(rows) == 2
    assert rows[0]["module"] == "model.router"
    assert rows[0]["arguments"]["weight"]["stride"] == [1, 8]
    assert rows[0]["arguments"]["weight"]["parameter"] == "model.router.weight"


def test_operator_wrapper_keeps_real_overload_and_schema_access():
    original = torch.ops.aten.add
    calls = []

    def invoke(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    observed = ObservedCallable(original, invoke)
    assert observed.default is original.default
    assert observed.Tensor is original.Tensor
    assert observed(torch.tensor(2), torch.tensor(3)).item() == 5
    assert len(calls) == 1


def test_norm_and_routing_arguments_keep_metadata_without_retaining_parameters():
    capture = observer()
    norm = torch.nn.LayerNorm(128, eps=1e-6, dtype=torch.float32)
    parameter = weakref.ref(norm.weight)
    capture.record("norm", {"normalization": norm}, None)
    del norm
    assert parameter() is None
    description = capture.snapshot()[0]["arguments"]["normalization"]
    assert description["parameters"]["weight"]["shape"] == [128]
    assert description["parameters"]["weight"]["dtype"] == "torch.float32"
    assert description["attributes"] == {"eps": 1e-6, "normalized_shape": [128]}

    @dataclasses.dataclass
    class Routing:
        top_k: int
        correction_bias: torch.Tensor

    routing = Routing(8, torch.ones(256))
    bias = weakref.ref(routing.correction_bias)
    capture.record("topk", {"routing": routing}, None)
    del routing
    assert bias() is None
    fields = capture.snapshot()[1]["arguments"]["routing"]["fields"]
    assert fields["top_k"] == 8 and fields["correction_bias"]["shape"] == [256]


def test_cache_location_snapshots_retain_small_decode_values_only():
    capture = observer(graph=7)
    locations = torch.tensor([[128], [0], [0]])
    capture.record("scatter", {"cache_locations": locations}, None)
    locations[0] = 129
    assert capture.snapshot(7)[0]["arguments"]["cache_locations"]["captured_values"] == [[129], [0], [0]]
    prefill = torch.ones((8192, 1), dtype=torch.int64)
    reference = weakref.ref(prefill)
    capture.record("scatter", {"cache_locations": prefill}, None)
    del prefill
    assert reference() is None
    assert "captured_values" not in capture.snapshot(7)[1]["arguments"]["cache_locations"]
