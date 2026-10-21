import copy
import gzip
import hashlib
import json

import pytest
from collector.sglang.ascend_memory_report import BOUNDARIES, summarize

pytestmark = pytest.mark.unit


@pytest.fixture
def observations(tmp_path):
    """Observed rank0 counters/KV geometry with a small synthetic aliased inventory."""
    source = tmp_path / "observer.py"
    source.write_text("observer fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    records = []
    allocated = [41220096, 52515494400, 52515494400, 58373111296, 58386254336, 58409285632]
    reserved = [44040192, 52915339264, 52915339264, 58766393344, 58766393344, 58969817088]
    free = [65228926976, 12351025152, 12351463424, 6497923072, 6497923072, 1910497280]
    for ordinal, boundary in enumerate(BOUNDARIES):
        records.append(
            {
                "schema": "aisimulate-ascend-worker-memory/v1",
                "boundary": boundary,
                "rank": 0,
                "device": 0,
                "time_ns": ordinal,
                "allocated_bytes": allocated[ordinal],
                "reserved_bytes": reserved[ordinal],
                "free_bytes": free[ordinal],
                "total_bytes": 65787658240,
                "sglang_commit": "baa9befa381511cd4d56f83980951e5a45fffc6d",
                "observer_sha256": digest,
            }
        )
    for ordinal in (1, 5):
        records[ordinal]["model_inventory"] = {
            "tensors": [
                {"name": "weight", "resident_npu": True, "storage_pointer": 1234, "storage_bytes": 64},
                {"name": "view", "resident_npu": True, "storage_pointer": 1234, "storage_bytes": 64},
            ],
            "unique_storages": 1,
            "unique_storage_bytes": 64,
        }
    records[3]["max_total_num_tokens"] = 52992
    records[3]["kv_pool_tensors"] = {
        "k_buffer": {"shape": [78, 415, 128, 1, 512], "dtype": "torch.bfloat16", "bytes": 4242800640},
        "v_buffer": {"shape": [78, 415, 128, 1, 64], "dtype": "torch.bfloat16", "bytes": 530350080},
        "index_k_buffer": {"shape": [78, 415, 128, 1, 128], "dtype": "torch.bfloat16", "bytes": 1060700160},
    }
    return tmp_path, source, records


def write_records(directory, records, compressed=False):
    payload = "".join(json.dumps(record) + "\n" for record in records).encode()
    if compressed:
        (directory / "rank-0.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))
    else:
        (directory / "rank-0.jsonl").write_bytes(payload)


@pytest.mark.parametrize("compressed", [False, True])
def test_reconciles_physical_pool_and_separate_allocator_domains(observations, compressed):
    directory, source, records = observations
    write_records(directory, records, compressed)
    report = summarize(directory, source, ranks=1)
    assert report["production_memory_model_ready"] is False
    rank = report["ranks"][0]
    assert rank["kv_storage_bytes"] == 5833850880
    assert rank["usable_tokens"] == 52992
    assert rank["phase_deltas"][2]["free_consumed_minus_reserved_delta_bytes"] == 4384002048
    assert rank["model_reachable_storage"]["load_model:after"]["unique_storage_bytes"] == 64


def test_rejects_missing_phase_and_rank(observations):
    directory, source, records = observations
    write_records(directory, records[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        summarize(directory, source, ranks=1)
    with pytest.raises(ValueError, match="every rank"):
        summarize(directory, source, ranks=2)


def test_rejects_compute_layer_count_as_storage_layer_count(observations):
    directory, source, records = observations
    records[3]["kv_pool_tensors"]["index_k_buffer"]["shape"][0] = 21
    write_records(directory, records)
    with pytest.raises(ValueError, match="index_k_buffer"):
        summarize(directory, source, ranks=1)


def test_rejects_changed_source_and_conflicting_alias(observations):
    directory, source, records = observations
    write_records(directory, records)
    original = source.read_text()
    source.write_text("changed observer")
    with pytest.raises(ValueError, match="source"):
        summarize(directory, source, ranks=1)
    source.write_text(original)
    changed = copy.deepcopy(records)
    changed[1]["model_inventory"]["tensors"][1]["storage_bytes"] = 128
    write_records(directory, changed)
    with pytest.raises(ValueError, match="Aliased"):
        summarize(directory, source, ranks=1)
