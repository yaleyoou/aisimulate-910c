import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    archive_path = root / "serving-controls.tar.gz"
    result = {
        "schema": "aisimulate-ascend-serving-control-report/v1",
        "status": "validated_measured_baseline",
        "prediction_accuracy_qualified": False,
        "timing_scope": "Client HTTP arrivals, including scheduling and frontend overhead; not device forward times",
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "cases": [],
    }
    with tarfile.open(archive_path, "r:gz") as archive:
        def read(name):
            member = archive.getmember(name)
            if not member.isfile():
                raise ValueError("Expected a regular evidence file")
            return archive.extractfile(member).read()

        for concurrency in (1, 10):
            prefix = f"controls-b{concurrency}"
            manifest = json.loads(read(prefix + "/manifest.json"))
            payload_bytes = read(prefix + "/requests.json")
            payloads = json.loads(payload_bytes)
            original = root.parent / "late-state-20261022/capture" / f"8192_1024_late_b{concurrency}/requests.json"
            assert payloads == json.loads(original.read_text())
            assert hashlib.sha256(payload_bytes).hexdigest() == manifest["requests_sha256"]
            assert hashlib.sha256((root / "ascend_serving_control.py").read_bytes()).hexdigest() == manifest["client_sha256"]
            assert manifest["status"] == "completed" and manifest["concurrency"] == concurrency
            assert len(manifest["trials"]) == manifest["warmup_trials"] + manifest["measurement_trials"] == 4
            trials = []
            for trial in manifest["trials"]:
                ordinal = trial["ordinal"]
                assert trial["warmup"] == (ordinal < manifest["warmup_trials"])
                observations = []
                for index in range(concurrency):
                    record = json.loads(read(f"{prefix}/trial-{ordinal}/request-{index}.json"))
                    metadata = record["response"]["meta_info"]
                    assert [metadata[key] for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "num_retractions")] == [8192, 1024, 0, 0]
                    arrivals = record["arrivals_tokens_ms"]
                    assert arrivals[-1][0] == 1024 and len(arrivals) > 1
                    assert all(math.isfinite(value) and value >= 0 for _, value in arrivals)
                    assert all(left[0] < right[0] and left[1] <= right[1] for left, right in zip(arrivals, arrivals[1:]))
                    assert record["ttft_ms"] == arrivals[0][1]
                    recomputed_tpot = (arrivals[-1][1] - arrivals[0][1]) / (arrivals[-1][0] - arrivals[0][0])
                    assert math.isclose(record["tpot_ms"], recomputed_tpot, rel_tol=1e-12)
                    for field in ("ttft_ms", "tpot_ms", "start_monotonic_ns", "end_monotonic_ns"):
                        assert record[field] == trial["requests"][index][field]
                    absolute = [(count, record["start_monotonic_ns"] / 1e6 + elapsed) for count, elapsed in arrivals]
                    observations.append({"record": record, "absolute": absolute})
                common_start = max(item["absolute"][0][1] for item in observations)
                common_end = min(item["absolute"][-1][1] for item in observations)
                late_start = max(next(elapsed for count, elapsed in item["absolute"] if count >= 900) for item in observations)
                windows = {}
                for name, lower in (("all_admitted", common_start), ("all_at_output900", late_start)):
                    intervals = []
                    for item in observations:
                        selected = [(count, elapsed) for count, elapsed in item["absolute"] if lower <= elapsed <= common_end]
                        if len(selected) < 2:
                            raise ValueError("No common steady decode window")
                        intervals.append((selected[-1][1] - selected[0][1]) / (selected[-1][0] - selected[0][0]))
                    windows[name] = {"duration_ms": common_end - lower, "mean_client_tpot_ms": statistics.mean(intervals)}
                recomputed_duration = (max(item["record"]["end_monotonic_ns"] for item in observations)
                    - min(item["record"]["start_monotonic_ns"] for item in observations)) / 1e9
                assert math.isclose(recomputed_duration, trial["duration_seconds"], rel_tol=1e-12)
                assert math.isclose(1024 * concurrency / recomputed_duration, trial["output_tokens_per_second"], rel_tol=1e-12)
                trials.append({
                    "ordinal": ordinal,
                    "warmup": trial["warmup"],
                    "mean_ttft_ms": statistics.mean(item["record"]["ttft_ms"] for item in observations),
                    "mean_tpot_ms": statistics.mean(item["record"]["tpot_ms"] for item in observations),
                    "output_tokens_per_second": trial["output_tokens_per_second"],
                    "common_windows": windows,
                })
            measurements = [trial for trial in trials if not trial["warmup"]]
            statistics_by_metric = {}
            for key in ("mean_ttft_ms", "mean_tpot_ms", "output_tokens_per_second"):
                values = [trial[key] for trial in measurements]
                statistics_by_metric[key] = {"mean": statistics.mean(values), "cv": statistics.pstdev(values) / statistics.mean(values)}
            result["cases"].append({"concurrency": concurrency, "matched_request_sha256": hashlib.sha256(original.read_bytes()).hexdigest(), "trials": trials, "measurement_statistics": statistics_by_metric})
    (root / "serving-controls-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": [{"concurrency": case["concurrency"], **case["measurement_statistics"]} for case in result["cases"]]}))


if __name__ == "__main__":
    main()
