import importlib
import io
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_prefill_capture_excludes_health_and_keeps_fast_own_requests(tmp_path, monkeypatch):
    package = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(package / "collector/sglang"))
    module = importlib.import_module("collector.sglang.ascend_state_sweep")
    control_path = tmp_path / "control.json"
    observed = []

    def request(_base, endpoint, _payload=None):
        assert json.loads(control_path.read_text())["enabled"] is False
        observed.append(endpoint)
        return {"tokens": [7, 8]} if endpoint == "/v1/tokenize" else {}

    def generate(_message, timeout):
        assert timeout == 600
        assert json.loads(control_path.read_text())["enabled"] is True
        record = {"meta_info": {"prompt_tokens": 8, "completion_tokens": 4, "cached_tokens": 0, "num_retractions": 0}}
        return io.BytesIO(("data: " + json.dumps(record) + "\n").encode())

    monkeypatch.setattr(module, "request", request)
    monkeypatch.setattr(module.urllib.request, "urlopen", generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep",
            "--capture-dir",
            str(tmp_path),
            "--case",
            "prefill",
            "--concurrency",
            "2",
            "--input-tokens",
            "8",
            "--output-tokens",
            "4",
            "--max-forwards",
            "1",
            "--capture-from-start",
        ],
    )
    module.main()
    assert observed[:2] == ["/health", "/flush_cache"]
    assert json.loads(control_path.read_text())["enabled"] is False
    result = json.loads((tmp_path / "prefill/workload-validation.json").read_text())
    assert result["status"] == "passed" and len(result["requests"]) == 2
