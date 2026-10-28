import argparse
import hashlib
import importlib
import json
from pathlib import Path

import pyarrow.parquet as parquet
from collector.sglang.ascend_curve_append import collection_identities, physical_coordinate, physical_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("bmm", "linear", "vector"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-memory-bytes", type=int, required=True)
    args = parser.parse_args()
    module = importlib.import_module("collector.sglang.ascend_" + args.operation + "_cases")
    kwargs = {"device_memory_bytes": args.device_memory_bytes} if args.operation == "bmm" else {}
    plan = getattr(module, "get_ascend_" + args.operation + "_cases")(args.model, **kwargs)
    table = args.baseline / ("ascend_" + args.operation + "_perf.parquet")
    previous = parquet.read_table(table).to_pylist()
    identities = collection_identities(json.loads((args.baseline / "metadata/provenance.json").read_text()))
    covered = {(physical_curve(args.operation, row), physical_coordinate(args.operation, row)) for row in previous}
    protocols = {}
    for row in previous:
        key = physical_curve(args.operation, row)
        protocol = identities[row["runtime_identity_sha256"]]["protocol"]
        if key in protocols and protocols[key] != protocol:
            raise ValueError("Existing curve mixes measurement protocols")
        protocols[key] = protocol
    groups = {}
    for case in plan["cases"]:
        physical = case
        if args.operation == "linear":
            physical = {**case, "weight_format": 29 if case["quant_mode"] in ("w8a8", "w8a8_dynamic") else 2}
        key = physical_curve(args.operation, physical)
        if (key, physical_coordinate(args.operation, case)) in covered:
            continue
        protocol = protocols.get(key, plan["protocol"])
        groups.setdefault(json.dumps(protocol, sort_keys=True), []).append(case)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for index, (protocol, cases) in enumerate(sorted(groups.items())):
        result = {
            **plan,
            "cases": cases,
            "protocol": json.loads(protocol),
            "selection": {
                "policy": "All new physical coordinates; retain each curve's original measurement protocol",
                "baseline_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                "generated": len(plan["cases"]),
                "selected": len(cases),
            },
        }
        path = args.output_dir / f"plan-{index}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "plan": str(path),
                    "cases": len(cases),
                    "protocol": result["protocol"],
                    "memory_exclusions": len(plan.get("memory_feasibility", {}).get("excluded", [])),
                }
            )
        )


if __name__ == "__main__":
    main()
