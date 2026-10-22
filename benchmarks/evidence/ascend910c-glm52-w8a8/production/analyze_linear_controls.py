"""Compare fixed-call cache controls and matched single-device work across nodes."""

import hashlib
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parent
    records = {}
    sources = {}
    for host in (31, 32):
        directory = root / f"linear-cache64-node{host}-20261022"
        records[host] = []
        for index, copies in enumerate((32, 64, 64, 32)):
            path = directory / f"m8-{index}-copies{copies}.json"
            content = path.read_bytes()
            sources[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
            record = json.loads(content)
            if record["status"] != "completed" or record["protocol"]["iterations"] != 128:
                raise ValueError("Incomplete or unmatched fixed-call experiment")
            records[host].append(record)
    reference = records[32][0]
    for runs in records.values():
        for record in runs:
            for key in ("framework_commit", "driver_version_info", "torch", "torch_npu", "device", "serving_setup"):
                if record["runtime"][key] != reference["runtime"][key]:
                    raise ValueError(f"Unmatched runtime: {key}")
            for key in ("plan_sha256", "collector_source_sha256", "selected_plan_indices"):
                if record[key] != reference[key]:
                    raise ValueError(f"Unmatched experiment: {key}")
    comparisons = []
    for offset, first in enumerate(reference["cases"]):
        entry = {"plan_index": first["plan_index"], "owners": first["owners"], "execution_mode": first["execution_mode"]}
        node_means = {}
        for host, runs in records.items():
            rows = [run["cases"][offset] for run in runs]
            if any(row["plan_index"] != first["plan_index"] for row in rows):
                raise ValueError("Mismatched physical case")
            values = [row["latency_ms"] for row in rows]
            mean32 = statistics.mean((values[0], values[3]))
            mean64 = statistics.mean(values[1:3])
            node_means[host] = mean64
            entry[str(host)] = {
                "latency_ms": values, "ratio_64_to_32": mean64 / mean32,
                "repeat_ratio_32": values[3] / values[0], "repeat_ratio_64": values[2] / values[1],
                "max_cv": max(row["cv"] for row in rows),
            }
        entry["node31_to_node32_at64"] = node_means[31] / node_means[32]
        comparisons.append(entry)
    summary = {
        "schema": "aisimulate-ascend-linear-controls/v1", "sources": sources, "cases": comparisons,
        "max_cross_node_relative_difference": max(abs(row["node31_to_node32_at64"] - 1) for row in comparisons),
        "max_cache_relative_difference": max(abs(row[str(host)]["ratio_64_to_32"] - 1) for row in comparisons for host in (31, 32)),
        "scope": "M8 controls for all declared matrices and both execution modes, two repeats per working set on each node.",
        "decision": "Use64weightcopies and128calls for the complete grid. These observations bound sampled node/cache sensitivity; they do not prove complete cache convergence or forward/serving accuracy. Keep all raw experiments.",
    }
    (root / "linear-controls-analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("sources", "cases")}))


if __name__ == "__main__":
    main()
