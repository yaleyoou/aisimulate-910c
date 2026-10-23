import hashlib
import json

import pytest

from collector.sglang.ascend_moe_chain_export import validated_rows

pytestmark = pytest.mark.unit


@pytest.fixture
def campaign(tmp_path):
    cases = [
        {
            "phase": "decode",
            "local_experts": 16,
            "hidden_size": 6144,
            "intermediate_size": 2048,
            "received_tokens": count,
            "physical_rows": 4096,
            "expert_counts": [count] + [0] * 15 if count == 1 else [1] * 16,
            "execution_mode": "graph",
            "group_list_type": 1,
            "distributions": ["balanced", "single_expert"] if count == 1 else ["balanced"],
        }
        for count in (1, 16)
    ]
    protocol = {"iterations": 8, "weight_copies": 8, "warmup": 5, "repeats": 3}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "aisimulate-ascend-moe-chain-plan/v1", "cases": cases, "protocol": protocol}))
    identity = dict.fromkeys(
        (
            "image_digest",
            "framework_commit",
            "driver_version_info",
            "hardware",
            "torch",
            "torch_npu",
            "collector_source_sha256",
        ),
        "fixture",
    )
    record = {
        "schema": "aisimulate-ascend-moe-chain-observations/v1",
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
                "weight_copies": 8,
                "weight_format": 29,
                "input_dtype": "int8",
                "output_dtype": "bfloat16",
                "scale_dtypes": ["float32", "bfloat16"],
                "kernel_sources": {
                    "gmm": "sglang.srt.hardware_backend.npu.moe.matmul.GroupedMatmul",
                    "activation": "sgl_kernel_npu.activation.swiglu_quant.swiglu_quant",
                },
                "execution_verification": "captured_graph_replay",
                "checks": [
                    {
                        "sampled_elements": 1024,
                        "max_abs_error": 0.01,
                        "max_quant_bin_error": 1,
                        "max_quant_scale_error": 0.0,
                    }
                ]
                * 2,
                "samples_ms": [float(index + 1)] * 3,
            }
            for index, case in enumerate(cases)
        ],
    }
    observation = tmp_path / "observations.json"
    observation.write_text(json.dumps(record))
    return plan, observation, identity, record


def test_equivalent_counts_reuse_measurement_without_losing_profile_identity(campaign):
    plan, observation, identity, _ = campaign
    rows, report = validated_rows(plan, [observation], identity)
    assert report["qualified_physical_cases"] == 2 and report["published_profile_rows"] == 3
    assert rows[0]["latency_ms"] == rows[1]["latency_ms"] == 1.0
    assert {rows[0]["distribution"], rows[1]["distribution"]} == {"balanced", "single_expert"}
    assert {row["physical_rows"] for row in rows} == {4096}


def test_frozen_count_vector_must_agree_with_its_profile(campaign):
    plan, observation, identity, record = campaign
    planned = json.loads(plan.read_text())
    planned["cases"][1]["expert_counts"] = [16] + [0] * 15
    record["cases"][1]["expert_counts"] = planned["cases"][1]["expert_counts"]
    plan.write_text(json.dumps(planned))
    record["plan_sha256"] = hashlib.sha256(plan.read_bytes()).hexdigest()
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="actual count vector"):
        validated_rows(plan, [observation], identity)


def test_wrong_quantization_and_missing_work_cannot_publish(campaign):
    plan, observation, identity, record = campaign
    record["cases"][0]["checks"][0]["max_quant_bin_error"] = 2
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="quantization check"):
        validated_rows(plan, [observation], identity)
    record["cases"][0]["checks"][0]["max_quant_bin_error"] = 1
    record["cases"].pop()
    record["selected_plan_indices"] = [0]
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="complete frozen plan"):
        validated_rows(plan, [observation], identity)


def test_unstable_case_does_not_generate_alias_rows(campaign):
    plan, observation, identity, record = campaign
    record["cases"][0]["samples_ms"] = [1.0, 1.0, 9.0]
    observation.write_text(json.dumps(record))
    rows, report = validated_rows(plan, [observation], identity)
    assert len(rows) == 1 and report["qualified_physical_cases"] == 1
    assert report["unqualified"][0]["plan_index"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        validated_rows(plan, [observation, observation], identity)
