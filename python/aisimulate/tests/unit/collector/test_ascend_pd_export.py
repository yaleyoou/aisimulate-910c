import hashlib
import json

import pytest
from collector.sglang.ascend_pd_export import validated_rows

pytestmark = pytest.mark.unit


@pytest.fixture
def campaign(tmp_path):
    case = {
        "source_tp": 8,
        "destination_tp": 4,
        "tokens": 128,
        "page_layout": "contiguous",
        "protocol": "sdma",
        "active_requests": 1,
        "ranks_per_node": 16,
        "num_layers": 78,
        "page_size": 128,
        "reference_pool_tokens": 53248,
        "component_widths": [512, 64, 128],
        "dtype": "bfloat16",
    }
    protocol = {"warmup": 3, "repeats": 3}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "aisimulate-ascend-pd-plan/v1", "cases": [case], "protocol": protocol}))
    plan_hash = hashlib.sha256(plan.read_bytes()).hexdigest()
    fields = ("torch", "torch_npu", "memfabric", "hardware", "driver")
    identity = dict.fromkeys((*fields, "framework_commit", "image_digest", "collector_source_sha256"), "fixture")
    mapping = [{"source_rank": rank * 2, "destination_rank": 16 + rank} for rank in range(4)]
    devices, results = [], []
    for rank in range(32):
        destinations = [entry["destination_rank"] for entry in mapping if entry["source_rank"] == rank]
        devices.append(
            {
                **dict.fromkeys(fields, "fixture"),
                "rank": rank,
                "local_rank": rank % 16,
                "host": "prefill" if rank < 16 else "decode",
                "cpu_affinity": [rank % 16],
                "source_sha256": "fixture",
                "plan_sha256": plan_hash,
                "selected_plan_indices": [0],
                "source_tp": 8,
                "destination_tp": 4,
                "pool_shapes": [[78, 417, 128, 1, width] for width in (512, 64, 128)],
                "pool_formats": [30] * 3,
            }
        )
        results.append(
            {
                "rank": rank,
                "destinations": destinations,
                "samples_ms": [float(rank + 1) if destinations else 0.0] * 3,
                "checks": [
                    {"sha256": str(nonce) * 64, "checked_bytes": 14_057_472 if 16 <= rank < 20 else 0}
                    for nonce in (1, 2)
                ],
            }
        )
    record = {
        "schema": "aisimulate-ascend-pd-observations/v1",
        "status": "completed",
        "negative_control": False,
        "plan_sha256": plan_hash,
        "collector_source_sha256": "fixture",
        "framework_commit": "fixture",
        "protocol": protocol,
        "serving_setup": {"worker_cpu_affinity": {str(rank): [rank] for rank in range(16)}},
        "devices": devices,
        "rank_mapping": mapping,
        "selected_plan_indices": [0],
        "cases": [
            {
                **case,
                "plan_index": 0,
                "source_pages": [1],
                "destination_pages": [2],
                "segments_per_buffer": 1,
                "payload_bytes_per_replica": 14_057_472,
                "total_wire_bytes": 56_229_888,
                "rank_results": results,
                "latency_ms": 7.0,
                "cv": 0.0,
                "kernel_source": "sglang.srt.disaggregation.ascend.conn.AscendKVManager.send_kvcache",
            }
        ],
    }
    observation = tmp_path / "observations.json"
    observation.write_text(json.dumps(record))
    return plan, observation, identity, record


def test_pd_uses_sender_critical_path_and_counts_receiver_replicas(campaign):
    plan, observation, identity, _ = campaign
    rows, report = validated_rows(plan, [observation], identity)
    assert report["qualified"] == 1
    assert rows[0]["latency_ms"] == 7.0
    assert rows[0]["total_wire_bytes"] == 56_229_888
    assert rows[0]["verified_receiver_copies"] == 8


@pytest.mark.parametrize("corruption", ("mapping", "digest", "missing_copy", "wire_bytes"))
def test_pd_must_verify_every_actual_receiver_copy(campaign, corruption):
    plan, observation, identity, record = campaign
    case = record["cases"][0]
    if corruption == "mapping":
        record["rank_mapping"][0]["source_rank"] = 1
    elif corruption == "digest":
        case["rank_results"][17]["checks"][1]["sha256"] = "3" * 64
    elif corruption == "missing_copy":
        case["rank_results"][17]["checks"][1]["checked_bytes"] = 0
    else:
        case["total_wire_bytes"] = case["payload_bytes_per_replica"]
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        validated_rows(plan, [observation], identity)
