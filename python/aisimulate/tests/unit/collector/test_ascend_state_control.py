import json

import pytest
from collector.sglang.ascend_state_capture import CaptureState

pytestmark = pytest.mark.unit


def test_capture_window_and_budget_are_per_case(tmp_path):
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"case_id": "decode", "enabled": True, "skip_forwards": 3, "max_forwards": 2}))
    state = CaptureState(tmp_path)
    observed = []
    for _ in range(7):
        enabled = state.control() is not None
        observed.append(enabled)
        if enabled:
            state.recorded += 1
    assert observed == [False, False, False, True, True, False, False]
    path.write_text(json.dumps({"case_id": "next", "enabled": True, "max_forwards": 1}))
    assert state.control() is not None
    assert state.forward_index == 0


def test_disabled_control_does_not_consume_capture_window(tmp_path):
    path = tmp_path / "control.json"
    state = CaptureState(tmp_path)
    assert state.control() is None
    path.write_text(json.dumps({"case_id": "admission", "enabled": False, "max_forwards": 2}))
    assert state.control() is None
    assert state.control() is None
    path.write_text(json.dumps({"case_id": "admission", "enabled": True, "max_forwards": 2}))
    assert state.control() is not None
    assert state.forward_index == 0


def test_invalid_case_path_is_rejected(tmp_path):
    (tmp_path / "control.json").write_text(json.dumps({"case_id": "../outside", "enabled": True, "max_forwards": 2}))
    with pytest.raises(ValueError, match="filename component"):
        CaptureState(tmp_path).control()
