import hashlib
import json

import pytest
from collector.sglang.ascend_linear_export import validated_rows

pytestmark = pytest.mark.unit


@pytest.fixture
def campaign(tmp_path):
    cases = [
        {
            "quant_mode": "w8a8",
            "m": 8,
            "n": 6144,
            "k": 2048,
            "quant_bias": bias,
            "execution_mode": "graph",
            "owners": [f"o.bias{bias}"],
        }
        for bias in (False, True)
    ]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"cases": cases}))
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
    protocol = {"weight_copies": 16, "iterations": 128, "warmup": 10, "repeats": 3}
    data = {
        "schema": "aisimulate-ascend-linear-observations/v1",
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
                "weight_format": 29,
                "input_dtype": "bfloat16",
                "output_dtype": "bfloat16",
                "kernel_source": (
                    "sglang.srt.hardware_backend.npu.quantization.linear_method_npu.NPUW8A8Int8LinearMethod"
                ),
                "execution_verification": "captured_graph_replay",
                "weight_copies": 16,
                "checks": [
                    {
                        "sampled_elements": 96,
                        "max_abs_error": 0.0,
                        "quant_max_bin_error": 0,
                        "quant_max_scale_error": 0.0,
                    }
                ]
                * 2,
                "samples_ms": [float(index + 1)] * 3,
            }
            for index, case in enumerate(cases)
        ],
    }
    observation = tmp_path / "observation.json"
    observation.write_text(json.dumps(data))
    return plan, observation, identity, data


def test_publishes_distinct_bias_rows_with_actual_execution_verification(campaign):
    plan, observation, identity, _ = campaign
    rows, report = validated_rows(plan, [observation], identity)
    assert report["attempted"] == report["qualified"] == 2
    assert [(row["quant_bias"], row["latency_ms"]) for row in rows] == [(False, 1.0), (True, 2.0)]
    assert {row["execution_verification"] for row in rows} == {"captured_graph_replay"}


def test_smoke_or_duplicate_work_cannot_be_published_as_full(campaign):
    plan, observation, identity, data = campaign
    with pytest.raises(ValueError, match="Duplicate"):
        validated_rows(plan, [observation, observation], identity)
    data["cases"] = data["cases"][:1]
    data["selected_plan_indices"] = [0]
    observation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="complete frozen plan"):
        validated_rows(plan, [observation], identity)


def test_unstable_rows_remain_reported_and_wrong_execution_fails(campaign):
    plan, observation, identity, data = campaign
    data["cases"][0]["samples_ms"] = [1.0, 1.0, 9.0]
    observation.write_text(json.dumps(data))
    rows, report = validated_rows(plan, [observation], identity)
    assert len(rows) == 1 and report["unqualified"][0]["plan_index"] == 0
    data["cases"][0]["execution_verification"] = "eager_invocation"
    observation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Actual execution"):
        validated_rows(plan, [observation], identity)


def test_quantization_reference_failure_cannot_pass_publication(campaign):
    plan, observation, identity, data = campaign
    data["cases"][0]["checks"][0]["quant_max_bin_error"] = 2
    observation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="numerical check"):
        validated_rows(plan, [observation], identity)


@pytest.mark.parametrize("field", ["driver_version_info", "torch", "torch_npu", "device", "framework_commit"])
def test_runtime_changes_require_a_distinct_campaign(campaign, field):
    plan, observation, identity, data = campaign
    data["runtime"][field] = "different"
    observation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Runtime"):
        validated_rows(plan, [observation], identity)
