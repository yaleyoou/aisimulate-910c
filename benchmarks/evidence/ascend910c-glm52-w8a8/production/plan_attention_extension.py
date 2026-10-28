import argparse
import hashlib
import json
from pathlib import Path

from collector.sglang.ascend_attention_cases import get_ascend_attention_cases
from collector.sglang.ascend_curve_append import physical_coordinate, physical_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--existing-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-heads", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--preset", choices=("baseline", "long-context"), default="baseline")
    args = parser.parse_args()
    original = get_ascend_attention_cases(args.model)
    previous = json.loads(args.existing_provenance.read_text())
    covered = {(tuple(curve["key"]), tuple(point)) for curve in previous["curves"] for point in curve["points"]}
    selected, retained, outside = [], 0, 0
    for case in original["cases"]:
        wanted = case["operation"] == "lightning_indexer" or case["local_heads"] in args.local_heads
        if case["phase"] == "prefill":
            wanted = wanted and (
                case["execution_mode"] == "eager"
                and case["active_batch"] in ((1, 2, 4) if args.preset == "long-context" else (1, 2))
                and case["tokens"] // case["active_batch"]
                in ((4096, 8192, 16384, 32768) if args.preset == "long-context" else (4096, 8192, 16384))
                and case["sequence_length"] <= (180000 if args.preset == "long-context" else 32768)
            )
        else:
            bucket = next(value for value in (8, 16, 24, 32) if value >= case["active_batch"])
            wanted = wanted and (
                case["execution_mode"] == "graph"
                and (args.preset == "long-context" or case["padded_batch"] == bucket)
                and case["sequence_length"]
                in (
                    (24576, 32768, 33792, 49152, 65536, 66560, 98304, 131072, 132096)
                    if args.preset == "long-context"
                    else (24576, 32768, 33792)
                )
            )
        if not wanted:
            outside += 1
        elif (physical_curve("attention", case), physical_coordinate("attention", case)) in covered:
            retained += 1
        else:
            selected.append(case)
    if not selected:
        raise ValueError("No new attention coordinates in the requested runtime subset")
    result = {
        **original,
        "cases": selected,
        "healing_selection": {
            "purpose": "Runtime coverage extension; new coordinates only, no timing-based selection",
            "preset": args.preset,
            "local_attention_heads": args.local_heads,
            "generated_cases": len(original["cases"]),
            "outside_requested_runtime_subset": outside,
            "existing_points_retained_without_remeasurement": retained,
            "new_cases": len(selected),
            "existing_provenance_sha256": hashlib.sha256(args.existing_provenance.read_bytes()).hexdigest(),
            "selector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["healing_selection"]))


if __name__ == "__main__":
    main()
