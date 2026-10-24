import argparse
import hashlib
import importlib
import json
from pathlib import Path

import pyarrow.parquet as parquet


def curve_key(operation, row):
    if operation == "linear":
        return tuple(row[field] for field in ("quant_mode", "n", "k", "quant_bias", "execution_mode"))
    dimensions = row["dimensions"]
    if isinstance(dimensions, str):
        dimensions = json.loads(dimensions)
    return row["role"], tuple(dimensions), row["execution_mode"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--operation", choices=("linear", "vector"), required=True)
    parser.add_argument("--baseline-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh immutable plan output")
    module = importlib.import_module("collector.sglang.ascend_" + args.operation + "_cases")
    plan = getattr(module, "get_ascend_" + args.operation + "_cases")(args.model)
    full_payload = json.dumps(plan, indent=2) + "\n"
    existing = {curve_key(args.operation, row) for row in parquet.read_table(args.baseline_table).to_pylist()}
    selected = [index for index, case in enumerate(plan["cases"]) if curve_key(args.operation, case) not in existing]
    if not selected:
        raise ValueError("No new physical curves in the declared plan")
    plan["selection"] = {
        "policy": "Runtime selection of entire previously unmeasured curves; no timing-dependent selection",
        "full_plan_sha256": hashlib.sha256(full_payload.encode()).hexdigest(),
        "baseline_table_sha256": hashlib.sha256(args.baseline_table.read_bytes()).hexdigest(),
        "full_plan_indices": selected,
    }
    plan["cases"] = [plan["cases"][index] for index in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".full.json").write_text(full_payload)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"operation": args.operation, "cases": len(selected), "curves": len({curve_key(args.operation, case) for case in plan["cases"]})}))


if __name__ == "__main__":
    main()
