import json

import pytest

from collector.sglang.ascend_hccl_export import validated_rows

pytestmark = pytest.mark.unit


def runtime():
    return dict.fromkeys(
        (
            "image_digest",
            "framework_commit",
            "cann_hccl_build",
            "driver_versions",
            "hardware",
            "collector_source_sha256",
        ),
        "test-fixture",
    )


def campaign(tmp_path, name, samples=None, iterations=16):
    path = tmp_path / name
    row = {
        "dtype": "bfloat16",
        "operation": "all_reduce",
        "execution_mode": "graph",
        "input_bytes_per_rank": 1024,
        "full_payload_correctness_checks": 4,
        "rank_samples": [{"device_ms": [1.0] * 3}, {"device_ms": samples or [2.0] * 3}],
    }
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "warmup": 10,
                "repeats": 3,
                "iterations": iterations,
                "schema": "aisimulate-hccl-observations/v2",
                "runtime_environment": {"TASK_QUEUE_ENABLE": "0", "HCCL_BUFFSIZE": "650"},
                "hccl_group_options": {"hccl_buffer_size": 650},
                "serving_setup": {
                    "schema": "sglang-runtime-setup/v1",
                    "environment": {"TASK_QUEUE_ENABLE": "0", "HCCL_BUFFSIZE": "650"},
                    "worker_cpu_affinity": {"0": [0], "1": [1]},
                },
                "num_ranks": 2,
                "ranks_per_node": 2,
                "rank_layout": "contiguous",
                "devices": [
                    {"rank": 0, "local_rank": 0, "cpu_affinity": [0]},
                    {"rank": 1, "local_rank": 1, "cpu_affinity": [1]},
                ],
                "cases": [row],
            }
        )
    )
    return path


def test_critical_rank_timing_and_exact_transport_identity(tmp_path):
    rows, report = validated_rows([campaign(tmp_path, "valid.json")], runtime())
    assert rows[0]["latency_ms"] == 2.0
    assert rows[0]["kernel_source"] == "torch.distributed.hccl"
    assert rows[0]["dtype"] == "bfloat16"
    assert rows[0]["correctness_checks"] == 4
    assert report["unqualified"] == []


def test_rejects_duplicates_and_protocol_mixing(tmp_path):
    first = campaign(tmp_path, "first.json")
    with pytest.raises(ValueError, match="Duplicate HCCL physical key"):
        validated_rows([first, first], runtime())
    second = campaign(tmp_path, "second.json", iterations=64)
    with pytest.raises(ValueError, match="different measurement protocols"):
        validated_rows([first, second], runtime())


def test_unstable_measurement_cannot_be_published(tmp_path):
    path = campaign(tmp_path, "unstable.json", samples=[2.0, 2.0, 10.0])
    with pytest.raises(ValueError, match="No qualified HCCL rows"):
        validated_rows([path], runtime())
    value = json.loads(path.read_text())
    value["cases"][0]["full_payload_correctness_checks"] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Missing pre/post correctness"):
        validated_rows([path], runtime())


def test_unrecorded_or_mismatched_runtime_options_cannot_be_published(tmp_path):
    path = campaign(tmp_path, "runtime.json")
    value = json.loads(path.read_text())
    value["hccl_group_options"]["hccl_buffer_size"] = 200
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="buffer differs"):
        validated_rows([path], runtime())
    value["schema"] = "aisimulate-hccl-observations/v1"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="requires v2"):
        validated_rows([path], runtime())
