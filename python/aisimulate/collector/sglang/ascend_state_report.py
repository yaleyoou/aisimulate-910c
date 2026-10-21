"""Validate complete MoE state captures and emit exact grouped-matmul cases."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def summarize(directory, ranks=16, layers=78, dense_layers=3, experts=256, expert_parallel=None):
    expert_parallel = ranks if expert_parallel is None else expert_parallel
    if expert_parallel < 1 or experts % expert_parallel or ranks % expert_parallel:
        raise ValueError("Invalid expert-parallel geometry")
    rank_records = {}
    for path in directory.glob("rank-*.jsonl"):
        rank = int(path.stem.split("-")[1])
        if rank in rank_records:
            raise ValueError("Duplicate rank file")
        rank_records[rank] = [json.loads(line) for line in path.read_text().splitlines()]
    if set(rank_records) != set(range(ranks)):
        raise ValueError("Capture must contain every expected rank")
    lengths = {len(records) for records in rank_records.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("Ranks have different or empty forward coverage")
    expected_layers = set(range(dense_layers, layers))
    cases = {}
    forward_summaries = []
    for ordinal in range(next(iter(lengths))):
        rows_per_layer = {}
        for rank, records in rank_records.items():
            record = records[ordinal]
            if record["rank"] != rank or record["ordinal"] != ordinal:
                raise ValueError("Rank/forward identity mismatch")
            seen = set()
            for entry in record["layers"]:
                match = re.search(r"layers\.(\d+)\.", entry["parameter"])
                if match is None:
                    raise ValueError("Parameter cannot be assigned to a model layer")
                layer = int(match.group(1))
                if layer in seen:
                    raise ValueError("Repeated layer invocation needs an explicit dimension")
                seen.add(layer)
                counts = entry["expert_counts"]
                if entry["group_list_type"] != 1 or len(counts) != experts // expert_parallel:
                    raise ValueError("Unexpected grouped-matmul expert/count layout")
                if entry["weight_shape"][0] != len(counts) or entry["weight_shape"][1] != entry["input_shape"][1]:
                    raise ValueError("Count/input/weight geometry mismatch")
                if any(type(value) is not int or value < 0 for value in counts):
                    raise ValueError("Expert counts must be nonnegative integers")
                if sum(counts) > entry["input_shape"][0]:
                    raise ValueError("Expert rows exceed the actual input storage")
                rows_per_layer.setdefault(layer, []).append(sum(counts))
                geometry = {
                    field: entry[field]
                    for field in (
                        "input_shape",
                        "input_dtype",
                        "weight_shape",
                        "weight_dtype",
                        "output_dtype",
                        "group_list_type",
                    )
                }
                geometry.update(expert_counts=counts, execution=record["execution"])
                encoded = json.dumps(geometry, sort_keys=True, separators=(",", ":"))
                key = hashlib.sha256(encoded.encode()).hexdigest()
                case = cases.setdefault(key, {"case_id": key, **geometry, "observations": []})
                case["observations"].append({"rank": rank, "layer": layer, "ordinal": ordinal})
            if seen != expected_layers:
                raise ValueError(f"Incomplete model-layer coverage on rank{rank}: {sorted(seen)}")
        forward_summaries.append(
            {
                "ordinal": ordinal,
                "per_rank_request_shape": [
                    {
                        "rank": rank,
                        "mode": records[ordinal]["forward_mode"],
                        "seq_lens": records[ordinal]["seq_lens"],
                        "input_tokens": records[ordinal]["input_tokens"],
                    }
                    for rank, records in sorted(rank_records.items())
                ],
                "group_rows_per_layer": {
                    str(layer): {"total": sum(values), "maximum_rank": max(values), "minimum_rank": min(values)}
                    for layer, values in sorted(rows_per_layer.items())
                },
            }
        )
    return {
        "schema": "aisimulate-ascend-moe-state-report/v1",
        "status": "validated",
        "scope": "observed group counts, not performance timings or logical-token estimates",
        "ranks": ranks,
        "expert_parallel": expert_parallel,
        "layers_per_forward": len(expected_layers),
        "forwards": forward_summaries,
        "unique_gmm1_cases": len(cases),
        "cases": sorted(cases.values(), key=lambda case: case["case_id"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directory)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"status": report["status"], "ranks": report["ranks"], "unique_gmm1_cases": report["unique_gmm1_cases"]}
        )
    )


if __name__ == "__main__":
    main()
