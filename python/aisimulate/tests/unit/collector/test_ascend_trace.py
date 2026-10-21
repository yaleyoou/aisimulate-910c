import csv
import io
import json
import tarfile

import pytest

from collector.sglang.ascend_trace import validate_response
from collector.sglang.ascend_trace_report import IDENTITY_FIELDS, read_archive, summarize

pytestmark = pytest.mark.unit


def archive_fixture(path, duration=10, rank=0, shape='"8,2048;2048,2048"'):
    fields = (*IDENTITY_FIELDS, "Duration(us)")
    row = dict.fromkeys(fields, "N/A")
    row.update(
        {
            "Device_id": str(rank),
            "Model ID": "46",
            "Name": "aclnnQuantMatmul",
            "Type": "QuantBatchMatmulV3",
            "Input Shapes": shape,
            "Input Data Types": "INT8;INT8",
            "Duration(us)": str(duration),
        }
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerow(row)
    entries = {
        f"rank-{rank}/ASCEND_PROFILER_OUTPUT/kernel_details.csv": output.getvalue(),
        f"rank-{rank}/ASCEND_PROFILER_OUTPUT/analyse.done": "",
        f"rank-{rank}/profiler_info_{rank}.json": json.dumps({"rank_id": rank}),
        f"rank-{rank}/ASCEND_PROFILER_OUTPUT/trace_view.json": json.dumps(
            [{"ph": "X", "name": "scheduler.run_batch", "ts": "1000", "dur": 20}]
        ),
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, value in entries.items():
            payload = value.encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_preserves_shape_dtype_rank_and_compact_evidence(tmp_path):
    path = tmp_path / "trace.tar.gz"
    archive_fixture(path)
    full = read_archive(path)
    compact = read_archive(tmp_path / "kernel-evidence.tar.gz")
    assert full == compact
    ranks, completed, kernels, spans = full
    assert ranks == {0}
    assert completed == {"rank-0"}
    identity = next(iter(kernels))
    assert identity[IDENTITY_FIELDS.index("Input Data Types")] == "INT8;INT8"
    assert identity[IDENTITY_FIELDS.index("Input Shapes")] == '"8,2048;2048,2048"'
    assert kernels[identity] == [10]
    assert len(spans) == 1


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_device_durations(tmp_path, duration):
    path = tmp_path / "trace.tar.gz"
    archive_fixture(path, duration=duration)
    with pytest.raises(ValueError, match="Invalid device duration"):
        read_archive(path)


def test_cached_or_retracted_workload_cannot_pass():
    expected = {"prompt_tokens": 8192, "completion_tokens": 1024, "cached_tokens": 0, "num_retractions": 0}
    assert validate_response({"meta_info": expected}, 8192, 1024) == expected
    for field in expected:
        invalid = {**expected, field: expected[field] + 1}
        with pytest.raises(ValueError, match=field):
            validate_response({"meta_info": invalid}, 8192, 1024)


def test_repeatability_rejects_jitter_and_missing_ranks(tmp_path):
    cases = []
    for index, duration in enumerate([10, 10, 40], start=1):
        directory = tmp_path / f"profile-{index}"
        directory.mkdir()
        archive_fixture(directory / "trace.tar.gz", duration=duration)
        cases.append({"path": directory.name})
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "collected_pending_trace_validation", "cases": cases})
    )
    with pytest.raises(ValueError, match="incomplete rank coverage"):
        summarize(tmp_path, expected_ranks=2)
    report = summarize(tmp_path, expected_ranks=1)
    assert report["repeatable_groups"] == 0
    assert report["unqualified_groups"] == 1
    assert report["canonical_fpm_data"] is False
