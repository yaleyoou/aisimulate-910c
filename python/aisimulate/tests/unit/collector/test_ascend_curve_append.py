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


def test_attention_grid_keeps_distinct_padding_and_prefix_coordinates():
    source = campaign(128, 2.0, {})
    source["rows"] = [
        {
            "operation": "sparse_attention",
            "phase": "decode",
            "local_heads": 32,
            "execution_mode": "graph",
            "active_batch": active,
            "padded_batch": padded,
            "sequence_length": 8192,
            "prefix": 0,
            "latency_ms": 1.0,
            "runtime_identity_sha256": source["rows"][0]["runtime_identity_sha256"],
        }
        for active, padded in [(1, 8), (1, 16), (2, 8)]
    ]
    rows, grids, counts = append_curves("attention", [source])
    assert len(rows) == 3 and len(grids) == 1 and counts == {0: 3}
    duplicate = copy.deepcopy(source)
    duplicate["rows"].append(dict(duplicate["rows"][0]))
    with pytest.raises(ValueError, match="duplicate points"):
        append_curves("attention", [duplicate])
