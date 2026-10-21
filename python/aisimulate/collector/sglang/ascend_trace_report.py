"""Validate and summarize experimental Ascend profiler archives without FPM publication."""

import argparse
import csv
import io
import json
import math
import statistics
import tarfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

IDENTITY_FIELDS = (
    "Device_id",
    "Model ID",
    "Name",
    "Type",
    "Accelerator Core",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
)


def read_archive(path):
    ranks = set()
    completed = set()
    kernels = defaultdict(list)
    spans = []
    with ExitStack() as stack:
        archive = stack.enter_context(tarfile.open(path, "r|gz"))
        compact = (
            stack.enter_context(tarfile.open(path.with_name("kernel-evidence.tar.gz"), "w:gz"))
            if path.name == "trace.tar.gz"
            else None
        )
        for member in archive:
            name = Path(member.name)
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            retained = name.name in ("kernel_details.csv", "analyse.done") or name.name.startswith("profiler_info_")
            if retained and compact is not None:
                content = stream.read()
                compact.addfile(member, io.BytesIO(content))
                stream = io.BytesIO(content)
            if name.name.startswith("profiler_info_"):
                ranks.add(json.load(stream)["rank_id"])
            elif name.name == "analyse.done":
                completed.add(name.parts[0])
            elif name.name == "kernel_details.csv":
                for row in csv.DictReader(io.StringIO(stream.read().decode("utf-8-sig"))):
                    identity = tuple(row[field] for field in IDENTITY_FIELDS)
                    duration = float(row["Duration(us)"])
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError(f"Invalid device duration: {duration}")
                    kernels[identity].append(duration)
            elif name.name == "trace_view.json":
                events = json.load(stream)
                if isinstance(events, dict):
                    events = events["traceEvents"]
                selected = [
                    {
                        "source": name.parts[0],
                        "name": event["name"],
                        "start_us": event["ts"],
                        "duration_us": event["dur"],
                    }
                    for event in events
                    if event.get("ph") == "X" and event.get("name") == "scheduler.run_batch"
                ]
                spans.extend(selected)
                if compact is not None:
                    payload = json.dumps(selected).encode()
                    info = tarfile.TarInfo(str(name.with_name("scheduler-spans.json")))
                    info.size = len(payload)
                    compact.addfile(info, io.BytesIO(payload))
            elif name.name == "scheduler-spans.json":
                spans.extend(json.load(stream))
    if not kernels:
        raise ValueError("Archive contains no device kernel observations")
    return ranks, completed, kernels, spans


def summarize(campaign, expected_ranks=16):
    manifest = json.loads((campaign / "manifest.json").read_text())
    if manifest["status"] != "collected_pending_trace_validation":
        raise ValueError(f"Collection did not complete: {manifest['status']}")
    profiles = [case for case in manifest["cases"] if case["path"].startswith("profile-")]
    if len(profiles) < 3:
        raise ValueError("Repeatability validation requires at least three independent profiles")
    groups = defaultdict(list)
    counts = []
    all_spans = []
    run_results = []
    for profile in profiles:
        case = profile["path"]
        path = campaign / case / "kernel-evidence.tar.gz"
        if not path.exists():
            path = campaign / case / "trace.tar.gz"
        ranks, completed, kernels, spans = read_archive(path)
        if ranks != set(range(expected_ranks)) or len(completed) != expected_ranks:
            raise ValueError(f"{case}: incomplete rank coverage: {sorted(ranks)}, exports={len(completed)}")
        per_type = Counter()
        for identity, samples in kernels.items():
            row = {field: identity[index] for index, field in enumerate(IDENTITY_FIELDS)}
            row["execution"] = "eager" if row["Model ID"] == "4294967295" else "graph"
            groups[identity].append({"case": case, "count": len(samples), "median_us": statistics.median(samples)})
            per_type[(row["Device_id"], row["execution"], row["Type"])] += len(samples)
        counts.extend(
            {"case": case, "device": device, "execution": execution, "type": op_type, "count": count}
            for (device, execution, op_type), count in sorted(per_type.items())
        )
        all_spans.extend({"case": case, **span} for span in spans)
        run_results.append(
            {
                "case": case,
                "ranks": sorted(ranks),
                "kernel_records": sum(map(len, kernels.values())),
                "scheduler_spans": len(spans),
            }
        )
    rows = []
    for identity, observations in sorted(groups.items()):
        medians = [observation["median_us"] for observation in observations]
        mean = statistics.mean(medians)
        cv = statistics.pstdev(medians) / mean if len(medians) > 1 else None
        repeatable = len(observations) == len(profiles) and cv <= 0.10
        rows.append(
            {
                **{field: identity[index] for index, field in enumerate(IDENTITY_FIELDS)},
                "independent_profiles": len(observations),
                "median_us": statistics.median(medians),
                "cv_of_profile_medians": cv,
                "repeatable_at_10_percent": repeatable,
                "observations": observations,
            }
        )
    report = {
        "schema": "aisimulate-ascend-trace-report/v1",
        "archive_integrity": "passed",
        "forward_boundary_validation": "requires_model_specific_operator_counts",
        "canonical_fpm_data": False,
        "runs": run_results,
        "kernel_groups": len(rows),
        "repeatable_groups": sum(row["repeatable_at_10_percent"] for row in rows),
        "unqualified_groups": sum(not row["repeatable_at_10_percent"] for row in rows),
        "limitations": [
            "Each group includes device rank and graph model ID; no pooling across ranks.",
            "Graph padding and MoE buffer capacity are physical shapes, not active token counts.",
            "Expert group-list contents and KV sequence lengths are not in kernel shape exports.",
            "Profiler perturbs timings; unqualified groups cannot be used as calibrated lookup rows.",
            "Communication summaries and their internal kernels overlap; never add both.",
            "Scheduler CPU spans are asynchronous launch intervals, not device forward latency.",
            "Serving correctness and stable output do not replace independent per-operator mathematical checks.",
        ],
    }
    for filename, value in (
        ("trace-report.json", report),
        ("kernel-groups.json", rows),
        ("kernel-counts.json", counts),
        ("scheduler-spans.json", all_spans),
    ):
        (campaign / filename).write_text(json.dumps(value, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--expected-ranks", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(summarize(args.campaign, args.expected_ranks), indent=2))


if __name__ == "__main__":
    main()
