import hashlib
import json

import pytest
from collector.sglang.ascend_pd_merge import select_curves


def campaign(lengths, latency, *, iterations=1):
    identity = {"serving_setup": {"affinity": "fixed"}, "protocol": {"iterations": iterations}}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "metadata": {"runtime": {"framework": "sglang", "version": "0.5.22"}},
        "report": {"identity": identity},
        "rows": [
            {
                "source_tp": 4,
                "destination_tp": 8,
                "page_layout": "fragmented",
                "tokens": count,
                "latency_ms": latency,
                "runtime_identity_sha256": digest,
            }
            for count in lengths
        ],
    }


def test_selection_preserves_a_whole_curve_and_ignores_faster_timing():
    existing = campaign([128, 8192, 16384], 100)
    fast_but_incomplete = campaign([128, 4096], 1, iterations=8)
    rows, decisions, counts = select_curves([existing, fast_but_incomplete])
    assert rows == existing["rows"]
    assert counts == {0: 3}
    assert decisions[0]["campaign"] == 0


def test_broader_curve_replaces_incomplete_coverage_without_splicing_points():
    old = campaign([128, 4096], 1)
    complete = campaign([1024, 8192, 16384], 100, iterations=8)
    rows, _, counts = select_curves([old, complete])
    assert rows == complete["rows"]
    assert 128 not in [row["tokens"] for row in rows]
    assert counts == {1: 3}


def test_merge_rejects_runtime_or_collection_identity_drift():
    first = campaign([128, 8192], 10)
    second = campaign([128, 8192], 10)
    second["metadata"]["runtime"]["version"] = "different"
    with pytest.raises(ValueError, match="different runtime"):
        select_curves([first, second])
    first["rows"][0]["runtime_identity_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="identity differ"):
        select_curves([first])
