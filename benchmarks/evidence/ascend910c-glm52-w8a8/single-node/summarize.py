import argparse
import csv
import json
import math
import pathlib
import statistics

from run import GROUPS, save


def summarize(root):
    plan = json.loads((root / "plan.json").read_text())
    reports = {name: json.loads((root / (name + ".json")).read_text()) for name in ["single", *GROUPS]}
    failures = []
    rows = []
    for name, report in reports.items():
        repeats = report["parameters"]["repeats"]
        expected = plan["expected_single_device_points"] if name == "single" else 20
        if not report["complete"] or report["errors"] or len(report["cases"]) != expected:
            failures.append({"report": name, "expected": expected, "actual": len(report["cases"]), "errors": report["errors"]})
        for index, case in enumerate(report["cases"]):
            timing = case["timing"]["device"] if name == "single" else case["timing"]
            samples = timing["samples_ms"]
            if len(samples) != repeats or not all(math.isfinite(value) and value > 0 for value in samples):
                failures.append({"report": name, "index": index, "error": "invalid samples"})
            if not math.isclose(statistics.median(samples), timing["median_ms"], rel_tol=1e-9):
                failures.append({"report": name, "index": index, "error": "incorrect median"})
            actual_cv = statistics.stdev(samples) / statistics.mean(samples)
            if not math.isclose(actual_cv, timing["cv"], rel_tol=1e-9, abs_tol=1e-12):
                failures.append({"report": name, "index": index, "error": "incorrect CV"})
            if name != "single":
                if case["devices"] != GROUPS[name]:
                    failures.append({"report": name, "index": index, "error": "incorrect rank placement"})
                rank_samples = case["rank_samples_ms"]
                if len(rank_samples) != len(GROUPS[name]) or any(len(values) != repeats for values in rank_samples):
                    failures.append({"report": name, "index": index, "error": "rank sample coverage"})
                for repeat, sample in enumerate(samples):
                    if sample != max(values[repeat] for values in rank_samples):
                        failures.append({"report": name, "index": index, "error": "incorrect critical rank timing"})
            rows.append({
                "report": name, "index": index, "operation": case["operation"],
                "device": case.get("device", ""), "precision": case.get("precision", case.get("dtype", "uint8")),
                "shape_mkn": str(case.get("shape_mkn", "")), "size_bytes": case.get("size_bytes", ""),
                "median_us": timing["median_ms"] * 1000, "cv": timing["cv"], "cv_above_10_percent": timing["cv"] > 0.1,
                "effective_gbps": case.get("effective_bytes_per_second", case.get("input_bytes_per_second", 0)) / 1e9,
                "effective_tops": case.get("effective_ops_per_second", 0) / 1e12,
            })
    if failures:
        save(root / "validation-errors.json", failures)
        raise ValueError(f"{len(failures)} collection integrity failures")
    with (root / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    local = reports["single"]
    calibration = {}
    for label, selected, value_key in [
        ("hbm_copy_read_plus_write_bytes_per_second", [case for case in local["cases"] if case["operation"] == "hbm_copy" and case["size_bytes"] == 512 << 20], "effective_bytes_per_second"),
        ("bf16_4096_cube_ops_per_second", [case for case in local["cases"] if case["operation"] == "matmul" and case["precision"] == "bf16" and case["shape_mkn"] == [4096, 4096, 4096]], "effective_ops_per_second"),
        ("int8_4096_cube_ops_per_second", [case for case in local["cases"] if case["operation"] == "matmul" and case["precision"] == "int8" and case["shape_mkn"] == [4096, 4096, 4096]], "effective_ops_per_second"),
    ]:
        if len(selected) != 16 or {case["device"] for case in selected} != set(range(16)):
            raise ValueError("Incomplete per-device calibration coverage: " + label)
        values = [case[value_key] for case in selected]
        calibration[label] = {
            "median_across_devices": statistics.median(values),
            "min_across_devices": min(values),
            "max_across_devices": max(values),
            "all_devices_repeat_cv_at_most_10_percent": all(case["timing"]["device"]["cv"] <= 0.1 for case in selected),
            "source": "single.json",
        }
    summary = {
        "scope": "observed single-node primitive performance, not theoretical hardware peaks or serving-operator performance",
        "execution_complete": True,
        "single_device_points": len(local["cases"]),
        "communication_points": sum(len(report["cases"]) for name, report in reports.items() if name != "single"),
        "sample_integrity_passed": True,
        "points_with_cv_above_10_percent": [row for row in rows if row["cv_above_10_percent"]],
        "stable_points": sum(not row["cv_above_10_percent"] for row in rows),
        "all_points_repeatability_passed": not any(row["cv_above_10_percent"] for row in rows),
        "local_source_reused": bool(plan.get("reused_local_source")),
        "local_source_sha256": plan.get("reused_local_source_sha256"),
        "nominal_runtime_device_count": 16,
        "minimum_runtime_total_memory_bytes": min(device["total_memory_bytes"] for device in local["devices"]),
        "calibration": calibration,
        "limitations": [
            "Local NPU-event measurements use eager Python submission; small kernels can include launch gaps.",
            "Communication times include eager host dispatch, enqueue, device work and final synchronization; they are not bare link latency.",
            "HCCL input bytes/second is not bus bandwidth; collective output sizes and algorithm traffic differ.",
            "INT8 timing excludes activation quantization and uses unit per-channel scales; no ModelSlim or fused MoE equivalence is asserted.",
            "Copy traffic counts one read plus one write; these are effective operation bytes, not hardware counters.",
            "No cross-node bandwidth, DeepEP dispatch/combine, DSA/indexer/MLA, complete forward or serving accuracy was measured.",
            "No production system YAML or AISimulate operator database was installed by this campaign.",
        ],
    }
    save(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    summary = summarize(args.output.resolve())
    print(json.dumps({key: value for key, value in summary.items() if key not in {"points_with_cv_above_10_percent", "limitations"}}, indent=2))


if __name__ == "__main__":
    main()
