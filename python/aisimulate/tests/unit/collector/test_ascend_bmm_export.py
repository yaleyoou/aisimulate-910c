import hashlib
import json

import pytest

from collector.sglang.ascend_bmm_export import validated_rows

pytestmark = pytest.mark.unit


@pytest.fixture
def campaign(tmp_path):
    cases = [
        {
            "role": "query_nope",
            "local_heads": 8,
            "tokens": tokens,
            "input_width": 192,
            "output_width": 512,
            "input_stride": [2048, 256, 1],
            "weight_stride": [98304, 1, 192],
            "execution_mode": "graph",
            "kernel_source": "torch_npu.npu_transpose_batchmatmul",
            "dtype": "bfloat16",
            "format": 2,
        }
        for tokens in (1, 9)
    ]
    protocol = {"weight_copies": 64, "iterations": 64, "warmup": 5, "repeats": 3}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "aisimulate-ascend-bmm-plan/v1", "cases": cases, "protocol": protocol}))
    identity = dict.fromkeys(
        (
            "image_digest",
            "framework_commit",
            "driver_version_info",
            "hardware",
            "collector_source_sha256",
            "torch",
            "torch_npu",
            "dispatch_source_sha256",
        ),
        "fixture",
    )
    record = {
        "schema": "aisimulate-ascend-bmm-observations/v1",
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
            "dispatch_source_sha256": "fixture",
            "device_index": 0,
            "cpu_affinity": [0],
            "serving_setup": {"worker_cpu_affinity": {"0": [0]}},
        },
        "selected_plan_indices": [0, 1],
        "cases": [
            {
                **case,
                "plan_index": index,
                "weight_copies": 64,
                "execution_verification": "captured_graph_replay",
                "checks": [{"sampled_elements": 24, "max_abs_error": 0.001}] * 2,
                "samples_ms": [float(index + 1)] * 3,
            }
            for index, case in enumerate(cases)
        ],
    }
    observation = tmp_path / "observation.json"
    observation.write_text(json.dumps(record))
    return plan, observation, identity, record


def test_publishes_only_verified_stride_contract(campaign):
    plan, observation, identity, _ = campaign
    rows, report = validated_rows(plan, [observation], identity)
    assert report["qualified"] == report["attempted"] == 2
    assert rows[0]["input_head_stride"] == 256
    assert rows[0]["weight_inner_stride"] == 1
    assert rows[0]["weight_output_stride"] == 192


def test_frozen_but_wrong_layout_cannot_be_published(campaign):
    plan, observation, identity, record = campaign
    planned = json.loads(plan.read_text())
    planned["cases"][0]["weight_stride"] = [98304, 512, 1]
    record["cases"][0]["weight_stride"] = [98304, 512, 1]
    plan.write_text(json.dumps(planned))
    record["plan_sha256"] = hashlib.sha256(plan.read_bytes()).hexdigest()
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="consumer contract"):
        validated_rows(plan, [observation], identity)


def test_unstable_and_incomplete_campaigns_remain_visible(campaign):
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


def test_unverified_graph_and_changed_dispatch_source_fail(campaign):
    plan, observation, identity, record = campaign
    record["cases"][0]["execution_verification"] = "eager_invocation"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Actual execution"):
        validated_rows(plan, [observation], identity)
    record["runtime"]["dispatch_source_sha256"] = "other"
    observation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Runtime"):
        validated_rows(plan, [observation], identity)
