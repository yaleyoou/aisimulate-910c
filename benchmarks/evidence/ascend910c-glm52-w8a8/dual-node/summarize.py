import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics


GROUPS = {"sio-2": 2, "hccs-2": 2, "local-16": 16, "cross-2": 2, "cross-32": 32}
SIZES = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304,
         16777216, 67108864, 134217728, 268435456]


def describe(samples):
    if not samples or not all(math.isfinite(value) and value > 0 for value in samples):
        raise ValueError("Missing or invalid timing samples")
    return {"median": statistics.median(samples), "minimum": min(samples), "maximum": max(samples),
            "cv": statistics.stdev(samples) / statistics.mean(samples) if len(samples) > 1 else None}


def summarize(output):
    manifest = json.loads((output / "manifest.json").read_text())
    queued = manifest.get("device_queue", False)
    failures = []
    groups = {}
    sources = {}
    total_samples = 0
    for label, ranks in GROUPS.items():
        points = []
        for rank in range(ranks):
            path = output / f"{label}-rank{rank}.jsonl"
            sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            metadata = [row for row in rows if row["type"] == "metadata"]
            completed = [row for row in rows if row["type"] == "complete"]
            assert len(metadata) == len(completed) == 1
            assert metadata[0]["rank"] == rank and metadata[0]["world"] == ranks
            assert metadata[0]["repeats"] == 9 and metadata[0]["iterations"] == 50
            expected_device = [0, 2][rank] if label == "hccs-2" else (0 if label == "cross-2" else rank % 16)
            assert metadata[0]["device"] == expected_device
            assert metadata[0]["peer"] == (rank + ranks // 2) % ranks
            if queued:
                assert metadata[0]["device_queue"] and not metadata[0]["capture"]
                checked = [row for row in rows if row["type"] == "post_timing_verification"]
                assert len(checked) == len(SIZES)
                assert {row["bytes"] for row in checked} == set(SIZES)
                assert all(row["correctness"] and row["rank"] == rank for row in checked)
            samples = [row for row in rows if row["type"] == "sample"]
            assert len(samples) == len(SIZES) * 9
            total_samples += len(samples)
            for size in SIZES:
                matched = [row for row in samples if row["bytes"] == size]
                assert len(matched) == 9 and {row["repeat"] for row in matched} == set(range(9))
                assert all(row["correctness"] and row["rank"] == rank for row in matched)
                device = describe([row["device_ms_per_direction"] for row in matched])
                wall = describe([row["wall_ms_per_direction"] for row in matched])
                if device["cv"] > 0.10:
                    failures.append({"group": label, "rank": rank, "bytes": size, "cv": device["cv"]})
                points.append({"rank": rank, "initiator": rank < ranks // 2, "bytes": size,
                               "device_ms_per_direction": device, "wall_ms_per_direction": wall})
        initiators = [point for point in points if point["initiator"]]
        by_size = {}
        for size in SIZES:
            matched = [point for point in initiators if point["bytes"] == size]
            times = [point["device_ms_per_direction"]["median"] for point in matched]
            by_size[str(size)] = {
                "median_ms_across_pairs": statistics.median(times),
                "min_ms_across_pairs": min(times), "max_ms_across_pairs": max(times),
                "maximum_repeat_cv": max(point["device_ms_per_direction"]["cv"] for point in matched),
                "median_effective_bytes_per_second": statistics.median(size / (value / 1000) for value in times),
            }
        groups[label] = {"ranks": ranks, "pairs": ranks // 2, "points": points, "by_size": by_size}
    intra = groups["local-16"]["by_size"][str(268435456)]["median_effective_bytes_per_second"]
    inter = groups["cross-32"]["by_size"][str(268435456)]["median_effective_bytes_per_second"]
    latency = max(statistics.median(groups[group]["by_size"][str(size)]["median_ms_across_pairs"]
                                   for size in [64, 256, 1024]) for group in ["local-16", "cross-32"]) / 1000
    validation = []
    for group, bandwidth in [("local-16", intra), ("cross-32", inter)]:
        for size in SIZES:
            if size in [64, 256, 1024, 268435456]:
                continue
            observed = groups[group]["by_size"][str(size)]["median_ms_across_pairs"] / 1000
            predicted = latency + size / bandwidth
            validation.append({"group": group, "bytes": size, "observed_seconds": observed,
                               "predicted_seconds": predicted, "relative_error": (predicted - observed) / observed})
    collective_points = []
    for path in sorted(output.glob("*-*-*.log")):
        if not any(operation in path.name for operation in ["all_reduce", "all_gather", "reduce_scatter", "alltoall"]):
            continue
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = re.findall(r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\w+)", path.read_text(), re.MULTILINE)
        assert len(rows) == 7, (path.name, rows)
        for size, micros, bandwidth, status in rows:
            assert status in {"success", "NULL"} and float(micros) > 0
            collective_points.append({"source": path.name, "bytes": int(size), "microseconds": float(micros),
                                      "algorithm_GB_per_second": float(bandwidth), "correctness": status})
    assert len(collective_points) == (14 if queued else 84), len(collective_points)
    calibration_failures = [item for item in failures if item["group"] in ["local-16", "cross-32"]
                            and item["rank"] < GROUPS[item["group"]] // 2
                            and item["bytes"] in [64, 256, 1024, 268435456]]
    return {
        "schema": "ascend910c-native-communication/v1", "source_sha256": sources,
        "p2p_samples": total_samples, "p2p_rank_size_points": total_samples // 9,
        "p2p_warmup_payload_checks": total_samples // 9,
        "correctness_scope": ("Full payload before AND after timed repeats at every rank/size." if queued else
                              "Full payload after warmup at every rank/size, before timed repeats.") + " Vendor collective check enabled; inspect each status.",
        "quality_limit_cv": 0.10, "cv_failures": failures, "calibration_cv_failures": calibration_failures,
        "groups": groups, "collective_points": collective_points,
        "collective_unverified_points": [point for point in collective_points if point["correctness"] != "success"],
        "system_calibration": {"intra_node_bw": round(intra), "inter_node_bw": round(inter), "p2p_latency": latency},
        "selection": "Median pair throughput at 256 MiB under concurrent pairs; maximum of local/cross small-payload medians for effective runtime P2P latency.",
        "held_out_payload_validation": validation,
        "held_out_max_absolute_relative_error": max(abs(row["relative_error"]) for row in validation),
        "canonical_inter_node_p2p_validation": [row for row in validation if row["group"] == "cross-32"],
        "canonical_inter_node_bulk_held_out_max_error": max(abs(row["relative_error"]) for row in validation
                                                            if row["group"] == "cross-32" and row["bytes"] >= 1048576),
        "canonical_inter_node_all_held_out_max_error": max(abs(row["relative_error"]) for row in validation
                                                           if row["group"] == "cross-32"),
        "timing_scope": ("ACL events after a host-callback gate; prequeue all HCCL calls and release after an MPI barrier. Device timing excludes enqueue gaps; wall timing includes queue preparation. Half round trip includes HCCL protocol, not bare-wire propagation." if queued else
                         "ACL events around native HCCL enqueue loop; half round trip includes launch gaps and runtime protocol, not bare fabric latency."),
        "scope": "Two Atlas 800T A3 nodes in SuperPod 1; 16 Ascend910_9382 runtime chips each; mixed host drivers.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    result = summarize(args.output)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["p2p_samples", "p2p_rank_size_points", "cv_failures",
          "calibration_cv_failures", "system_calibration", "held_out_max_absolute_relative_error"]}, indent=2))


if __name__ == "__main__":
    main()
