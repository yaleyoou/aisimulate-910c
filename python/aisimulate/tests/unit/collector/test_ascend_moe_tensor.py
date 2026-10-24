import copy
import hashlib
import json
from pathlib import Path

import pytest
from collector.sglang.ascend_moe_tensor_cases import get_ascend_moe_tensor_cases
from collector.sglang.ascend_moe_tensor_export import KERNEL_SOURCES, validated_rows

pytestmark = pytest.mark.unit


def model_files(tmp_path):
    config = Path(__file__).resolve().parents[3] / "src/aisimulate_core/model_configs/zai-org--GLM-5.2_config.json"
    (tmp_path / "config.json").write_bytes(config.read_bytes())
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps({"model.layers.3.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC"})
    )
    return tmp_path


def test_tensor_shapes_follow_the_model_shard_and_preserve_graph_modes(tmp_path):
    plan = get_ascend_moe_tensor_cases(model_files(tmp_path))
    assert len(plan["cases"]) == 128
    assert {case["intermediate_size"] for case in plan["cases"]} == {64, 128, 256, 512}
    assert {case["num_experts"] for case in plan["cases"]} == {256}
    for width in (64, 128, 256, 512):
        for mode in ("eager", "graph"):
            counts = {
                case["tokens"]
                for case in plan["cases"]
                if case["intermediate_size"] == width and case["execution_mode"] == mode
            }
            assert {1, 8, 16, 24, 32, 8192, 16384} <= counts
    (tmp_path / "quant_model_description.json").write_text("{}")
    with pytest.raises(ValueError, match="ModelSlim"):
        get_ascend_moe_tensor_cases(tmp_path)


def fixture(tmp_path):
    plan = get_ascend_moe_tensor_cases(model_files(tmp_path))
    plan["cases"] = plan["cases"][:2]
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
    setup = {"worker_cpu_affinity": {"0": [0]}, "environment": {}}
    record = {
        "schema": "aisimulate-ascend-moe-tensor-observations/v1",
        "status": "completed",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "collector_source_sha256": identity["collector_source_sha256"],
        "protocol": plan["protocol"],
        "runtime": {
            **identity,
            "device": identity["hardware"],
            "device_index": 0,
            "serving_setup": setup,
            "cpu_affinity": [0],
        },
        "selected_plan_indices": [0, 1],
        "cases": [],
    }
    for index, case in enumerate(plan["cases"]):
        record["cases"].append(
            {
                **case,
                "plan_index": index,
                "status": "passed",
                "weight_format": 29,
                "input_dtype": "bfloat16",
                "output_dtype": "bfloat16",
                "group_list_type": 1,
                "quant_mode": 1,
                "drop_pad_mode": 2,
                "weight_copies": 8,
                "kernel_sources": KERNEL_SOURCES,
                "execution_verification": "eager_invocation",
                "samples_ms": [1.0] * 7,
                "wall_samples_ms": [1.1] * 7,
                "checks": [
                    {"sampled_elements": 1000, "max_abs_error": 0.01, "input_mutation": sign} for sign in (-1, 1)
                ],
            }
        )
    return plan_path, identity, record


def test_export_requires_all_cases_and_preserves_unstable_measurements(tmp_path):
    plan, identity, record = fixture(tmp_path)
    record["cases"][1]["samples_ms"] = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]
    observation = tmp_path / "observed.json"
    observation.write_text(json.dumps(record))
    rows, report = validated_rows(plan, [observation], identity)
    assert len(rows) == 1 and rows[0]["latency_ms"] == 1.0
    assert report["attempted_physical_cases"] == 2 and report["unqualified"][0]["plan_index"] == 1
    record["cases"].pop()
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Missing planned"):
        validated_rows(plan, [observation], identity)


@pytest.mark.parametrize("change", ["kernel", "mutation", "protocol", "duplicate"])
def test_export_rejects_wrong_execution_identity(tmp_path, change):
    plan, identity, record = fixture(tmp_path)
    record = copy.deepcopy(record)
    if change == "kernel":
        record["cases"][0]["kernel_sources"][2] = "NPUSwigluDeepEPKernel"
    elif change == "mutation":
        record["cases"][0]["checks"][0]["input_mutation"] = 1
    elif change == "protocol":
        record["protocol"]["iterations"] *= 2
    else:
        record["selected_plan_indices"][1] = record["cases"][1]["plan_index"] = 0
    observation = tmp_path / "observed.json"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        validated_rows(plan, [observation], identity)
