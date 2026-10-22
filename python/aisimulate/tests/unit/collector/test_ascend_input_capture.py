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
