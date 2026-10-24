import copy
import hashlib
import json

import pytest
from collector.sglang.ascend_curve_append import append_curves

pytestmark = pytest.mark.unit


def campaign(width, latency, protocol):
    identity = {"runtime": {"version": "0.5.22"}, "protocol": protocol, "serving_setup": {"environment": {}}}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "report": {"identity": identity},
        "metadata": {"runtime": identity["runtime"]},
        "rows": [
            {
                "role": "swiglu",
                "dimensions": json.dumps([width]),
                "execution_mode": "graph",
                "tokens": tokens,
                "latency_ms": latency,
                "runtime_identity_sha256": digest,
            }
            for tokens in (8, 16)
        ],
    }


def test_disjoint_curves_keep_original_timings_and_collection_protocols():
    first, second = campaign(128, 2.0, {"iterations": 8}), campaign(256, 1.0, {"iterations": 16})
    rows, curves, counts = append_curves("vector", [first, second])
    assert rows == first["rows"] + second["rows"]
    assert len(curves) == 2 and counts == {0: 2, 1: 2}
    assert curves[0]["collection_identity_sha256"] != curves[1]["collection_identity_sha256"]


def test_overlapping_curve_cannot_replace_slower_points():
    with pytest.raises(ValueError, match="Overlapping curves"):
        append_curves("vector", [campaign(128, 2.0, {}), campaign(128, 1.0, {})])


def test_unverified_identity_and_different_serving_setup_are_rejected():
    first, second = campaign(128, 2.0, {}), campaign(256, 1.0, {})
    invalid = copy.deepcopy(second)
    invalid["rows"][0]["runtime_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="verified collection"):
        append_curves("vector", [first, invalid])
    invalid = copy.deepcopy(second)
    invalid["report"]["identity"]["serving_setup"]["environment"]["different"] = "1"
    with pytest.raises(ValueError, match="serving setups"):
        append_curves("vector", [first, invalid])
