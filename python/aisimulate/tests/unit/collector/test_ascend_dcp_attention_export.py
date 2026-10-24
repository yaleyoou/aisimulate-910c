import copy
import hashlib
import json

import pytest
from collector.sglang.ascend_dcp_attention_export import CONTRACT, validated_rows

pytestmark = pytest.mark.unit


def fixture(tmp_path):
    cases = [
        {
            "local_heads": 8,
            "dcp_size": 2,
            "remap_topk": True,
            "page_size": 128,
            "sparse_topk": 2048,
            "kv_lora_rank": 512,
            "rope_dim": 64,
            "kv_interleave_size": 128,
            "phase": "decode",
            "execution_mode": "graph",
            "active_batch": 1,
            "padded_batch": 8,
            "query_tokens_per_request": 1,
            "tokens": 8,
            "sequence_length": length,
            "prefix": 0,
            "pool_pages": 209,
            "block_table_width": 704,
        }
        for length in (8192, 9216)
    ]
    protocol = {"iterations": 8, "warmup": 5, "repeats": 7}
    plan = {
        "schema": "aisimulate-ascend-dcp-attention-plan/v1",
        "boundary_contract": CONTRACT,
        "cases": cases,
        "protocol": protocol,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    identity = {
        "image_digest": "sha256:" + "1" * 64,
        "framework_commit": "2" * 40,
        "driver_version_info": "driver",
        "hardware": "Ascend910_9391",
        "collector_source_sha256": "3" * 64,
        "torch": "2.10.0",
        "torch_npu": "2.10.0",
    }
    workers = [
        {
            "rank": rank,
            "local_rank": rank,
            "host": "one-node",
            "device": identity["hardware"],
            "torch": identity["torch"],
            "torch_npu": identity["torch_npu"],
            "driver_version_info": "driver",
            "cpu_affinity": [rank],
        }
        for rank in (0, 1)
    ]
    record = {
        "schema": "aisimulate-ascend-dcp-attention-observations/v1",
        "status": "completed",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "collector_source_sha256": identity["collector_source_sha256"],
        "framework_commit": identity["framework_commit"],
        "protocol": protocol,
        "negative_control": False,
        "serving_setup": {"worker_cpu_affinity": {"0": [0], "1": [1]}, "environment": {}},
        "runtime": workers,
        "selected_plan_indices": [0, 1],
        "cases": [],
    }
    for index, case in enumerate(cases):
        ranks = [
            {
                "rank": rank,
                "samples_ms": [1.0] * 7,
                "wall_samples_ms": [1.1] * 7,
                "checks": [
                    {
                        "rank": rank,
                        "sampled_elements": 1000,
                        "max_abs_error": 0.001,
                        "max_lse_error": 0.00001,
                        "ideal_fp32_max_abs_error": 0.005,
                        "input_mutation": sign,
                    }
                    for sign in (-1, 1)
                ],
            }
            for rank in (0, 1)
        ]
        record["cases"].append(
            {
                **case,
                "plan_index": index,
                "status": "passed",
                "kernel_contract": CONTRACT,
                "cache_format": 30,
                "query_dtype": "bfloat16",
                "output_dtype": "bfloat16",
                "merge_dtype": "float32",
                "group_ranks": [0, 1],
                "rank_layout": "contiguous",
                "ranks_per_node": 2,
                "local_cache_index_dtype": "int32",
                "indexer_cache_index_dtype": "int64",
                "execution_verification": "captured_graph_replay",
                "rank_measurements": ranks,
                "samples_ms": [1.0] * 7,
                "latency_ms": 1.0,
                "cv": 0.0,
            }
        )
    directory = tmp_path / "campaign"
    directory.mkdir()
    manifest = {
        "schema": "aisimulate-ascend-dcp-campaign/v1",
        "status": "completed",
        "negative_control": False,
        "contamination": [],
        "planned_indices": [0, 1],
        "attempted_cases": 2,
        "failed_cases": [],
        "attempts": [{"observation": "observations-0000.json", "completed_indices": [0, 1], "returncode": 0}],
    }
    return plan_path, directory, identity, record, manifest


def save(directory, record, manifest):
    (directory / "observations-0000.json").write_text(json.dumps(record))
    (directory / "campaign.json").write_text(json.dumps(manifest))


def test_exact_rank_maximum_and_immutable_identity_are_published(tmp_path):
    plan, directory, identity, record, manifest = fixture(tmp_path)
    save(directory, record, manifest)
    rows, report = validated_rows(plan, [directory], identity)
    assert len(rows) == report["attempted"] == report["qualified"] == 2
    assert {row["latency_ms"] for row in rows} == {1.0}
    assert len({row["runtime_identity_sha256"] for row in rows}) == 1
    assert not report["unqualified"]


def test_numerical_failure_remains_excluded_without_losing_plan_coverage(tmp_path):
    plan, directory, identity, record, manifest = fixture(tmp_path)
    failed = record["cases"].pop()
    record.update(status="failed", failed_plan_index=1)
    manifest["attempts"][0].update(completed_indices=[0], returncode=1)
    manifest["failed_cases"] = [
        {
            "plan_index": 1,
            "classification": "numerical_check_failed",
            "observation": "observations-0000.json",
            "failure_files": ["rank-failure.json"],
        }
    ]
    case = json.loads(plan.read_text())["cases"][1]
    (directory / "rank-failure.json").write_text(
        json.dumps(
            {
                "plan_index": 1,
                "case": case,
                "collector_source_sha256": identity["collector_source_sha256"],
                "error": "DCP execution numerical check failed: partial attention",
            }
        )
    )
    save(directory, record, manifest)
    rows, report = validated_rows(plan, [directory], identity)
    assert len(rows) == 1 and report["attempted"] == 2
    assert report["unqualified"][0]["reason"] == "numerical_check_failed"
    assert failed["sequence_length"] == 9216


@pytest.mark.parametrize(
    "field,value",
    [
        ("kv_interleave_size", 1),
        ("indexer_cache_index_dtype", "int32"),
        ("kernel_contract", "sglang_npu_dcp_sparse_chain/v1"),
        ("samples_ms", [0.1] * 7),
    ],
)
def test_wrong_layout_boundary_or_faster_aggregate_is_rejected(tmp_path, field, value):
    plan, directory, identity, record, manifest = fixture(tmp_path)
    record["cases"][0][field] = value
    save(directory, record, manifest)
    with pytest.raises(ValueError):
        validated_rows(plan, [directory], identity)


def test_duplicate_campaign_cannot_choose_between_repeated_measurements(tmp_path):
    plan, directory, identity, record, manifest = fixture(tmp_path)
    save(directory, record, manifest)
    with pytest.raises(ValueError, match="repeated"):
        validated_rows(plan, [directory, directory], identity)
    invalid = copy.deepcopy(record)
    invalid["cases"][0]["rank_measurements"].pop()
    save(directory, invalid, manifest)
    with pytest.raises(ValueError, match="rank measurements"):
        validated_rows(plan, [directory], identity)


def test_audited_stop_and_disjoint_continuation_must_cover_the_original_plan(tmp_path):
    plan, directory, identity, record, manifest = fixture(tmp_path)
    continuation_record = copy.deepcopy(record)
    continuation_manifest = copy.deepcopy(manifest)
    record.update(status="failed", failed_plan_index=0, cases=[])
    manifest.update(
        status="failed",
        attempted_cases=1,
        error=("RuntimeError: DCP numerical failures reached the investigation threshold; remaining cases untouched"),
    )
    manifest["attempts"][0].update(completed_indices=[], returncode=1)
    manifest["failed_cases"] = [
        {
            "plan_index": 0,
            "classification": "numerical_check_failed",
            "observation": "observations-0000.json",
            "failure_files": ["rank-failure.json"],
        }
    ]
    (directory / "rank-failure.json").write_text(
        json.dumps(
            {
                "plan_index": 0,
                "case": json.loads(plan.read_text())["cases"][0],
                "collector_source_sha256": identity["collector_source_sha256"],
                "error": "DCP execution numerical check failed: partial attention",
            }
        )
    )
    save(directory, record, manifest)
    with pytest.raises(ValueError, match="recorded investigation"):
        validated_rows(plan, [directory], identity)
    (directory / "investigation.json").write_text(
        json.dumps({"status": "audited_numerical_exclusions", "affected_indices": [0]})
    )
    with pytest.raises(ValueError, match="every planned case"):
        validated_rows(plan, [directory], identity)
    continuation = tmp_path / "continuation"
    continuation.mkdir()
    continuation_record["selected_plan_indices"] = [1]
    continuation_record["cases"] = continuation_record["cases"][1:]
    continuation_manifest.update(planned_indices=[1], attempted_cases=1)
    continuation_manifest["attempts"][0]["completed_indices"] = [1]
    save(continuation, continuation_record, continuation_manifest)
    rows, report = validated_rows(plan, [directory, continuation], identity)
    assert report["attempted"] == 2 and report["qualified"] == 1
    assert rows[0]["sequence_length"] == 9216
