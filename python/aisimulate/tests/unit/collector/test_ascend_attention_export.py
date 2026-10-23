import hashlib
import json

import pytest

from collector.sglang.ascend_attention_export import validated_rows

pytestmark = pytest.mark.unit


@pytest.fixture
def campaign(tmp_path):
    cases = [
        dict(
            operation="sparse_attention",
            phase="decode",
            local_heads=8,
            tokens=8,
            active_batch=active,
            padded_batch=8,
            sequence_length=8192,
            prefix=0,
            execution_mode="graph",
            page_size=128,
            sparse_topk=2048,
            kv_lora_rank=512,
            rope_dim=64,
            index_heads=32,
            index_dim=128,
            pool_pages=417,
            block_table_width=1407,
        )
        for active in (1, 5)
    ]
    protocol = {"iterations": 8, "warmup": 5, "repeats": 3}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "aisimulate-ascend-attention-plan/v1", "cases": cases, "protocol": protocol}))
    identity = dict.fromkeys(
        (
            "image_digest",
            "framework_commit",
            "driver_version_info",
            "hardware",
            "collector_source_sha256",
            "torch",
            "torch_npu",
        ),
        "fixture",
    )
    record = {
        "schema": "aisimulate-ascend-attention-observations/v1",
        "status": "completed",
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "collector_source_sha256": "fixture",
        "protocol": protocol,
        "runtime": {
            "framework_commit": "fixture",
            "driver_version_info": "fixture",
            "device": "fixture",
            "torch": "fixture",
            "torch_npu": "fixture",
            "device_index": 0,
            "cpu_affinity": [0],
            "serving_setup": {"worker_cpu_affinity": {"0": [0]}},
        },
        "selected_plan_indices": [0, 1],
        "cases": [
            {
                **case,
                "plan_index": index,
                "cache_format": 30,
                "cache_stride": [65536, 512, 512, 1],
                "kernel_source": "torch_npu.npu_sparse_flash_attention",
                "execution_verification": "captured_graph_replay",
                "checks": [
                    {
                        "sampled_queries": 24,
                        "max_abs_error": 0.001,
                        "minimum_topk_overlap": 1.0,
                        "padding_kv_lengths": [0] * (8 - case["active_batch"]),
                    }
                    for _ in range(2)
                ],
                "samples_ms": [float(index + 1)] * 3,
            }
            for index, case in enumerate(cases)
        ],
    }
    observation = tmp_path / "observation.json"
    observation.write_text(json.dumps(record))
    return plan, observation, identity, record


def test_padded_rows_do_not_replace_active_batch(campaign):
    plan, observation, identity, _ = campaign
    rows, report = validated_rows(plan, [observation], identity)
    assert report["qualified"] == report["attempted"] == 2
    assert [(row["active_batch"], row["padded_batch"], row["latency_ms"]) for row in rows] == [(1, 8, 1.0), (5, 8, 2.0)]


@pytest.mark.parametrize(
    "field,value", [("cache_format", 2), ("cache_stride", [65536, 512, 1, 1]), ("kernel_source", "other")]
)
def test_actual_layout_and_kernel_must_match(campaign, field, value):
    plan, observation, identity, record = campaign
    record["cases"][0][field] = value
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="consumer contract"):
        validated_rows(plan, [observation], identity)


def test_unstable_samples_remain_in_report_and_partial_campaigns_fail(campaign):
    plan, observation, identity, record = campaign
    record["cases"][0]["samples_ms"] = [1.0, 1.0, 9.0]
    observation.write_text(json.dumps(record))
    rows, report = validated_rows(plan, [observation], identity)
    assert len(rows) == 1 and report["unqualified"][0]["plan_index"] == 0
    record["cases"].pop()
    record["selected_plan_indices"] = [0]
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="complete frozen plan"):
        validated_rows(plan, [observation], identity)


def test_nonzero_padding_and_unverified_graph_are_rejected(campaign):
    plan, observation, identity, record = campaign
    record["cases"][0]["checks"][0]["padding_kv_lengths"][0] = 1
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="padding"):
        validated_rows(plan, [observation], identity)
    record["cases"][0]["execution_verification"] = "eager_invocation"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Actual execution"):
        validated_rows(plan, [observation], identity)


def test_mixed_runtime_and_duplicate_cases_fail(campaign):
    plan, observation, identity, record = campaign
    with pytest.raises(ValueError, match="Duplicate"):
        validated_rows(plan, [observation, observation], identity)
    record["runtime"]["driver_version_info"] = "other"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Runtime"):
        validated_rows(plan, [observation], identity)


def test_recorded_numerical_failure_remains_visible_without_a_timing_row(campaign):
    plan, observation, identity, record = campaign
    failed = record["cases"].pop()
    record["failures"] = [
        {
            "plan_index": failed["plan_index"],
            "case": json.loads(plan.read_text())["cases"][1],
            "classification": "unexpected",
            "exception_type": "ValueError",
            "error": "Sparse attention execution numerical check failed: actual0",
        }
    ]
    observation.write_text(json.dumps(record))
    rows, report = validated_rows(plan, [observation], identity)
    assert len(rows) == 1 and report["attempted"] == 2
    assert report["unqualified"][0]["reason"] == "numerical execution check failed"
    record["failures"][0]["error"] = "Skipped without execution"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="failure record"):
        validated_rows(plan, [observation], identity)
