"""Freeze whole PD curves lacking8K coverage under an explicit longer sampling protocol."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("observations", type=Path, nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh supplemental plan")
    plan = json.loads(args.plan.read_text())
    seen, groups = set(), defaultdict(list)
    for path in args.observations:
        record = json.loads(path.read_text())
        if record["status"] != "completed" or record["plan_sha256"] != hashlib.sha256(args.plan.read_bytes()).hexdigest():
            raise ValueError("Supplement decisions require the complete original collection")
        for row in record["cases"]:
            if row["plan_index"] in seen:
                raise ValueError("Duplicate original point")
            seen.add(row["plan_index"])
            if row["cv"] <= 0.1:
                groups[row["source_tp"], row["destination_tp"], row["page_layout"]].append(row["tokens"])
    if seen != set(range(len(plan["cases"]))):
        raise ValueError("Original collection is incomplete")
    keys = {(case["source_tp"], case["destination_tp"], case["page_layout"]) for case in plan["cases"]}
    missing = {key for key in keys if not groups[key] or min(groups[key]) > 8192 or max(groups[key]) < 8192}
    if not missing:
        print(json.dumps({"supplement_required": False}))
        return
    original_protocol = dict(plan["protocol"])
    plan["cases"] = [case for case in plan["cases"] if (case["source_tp"], case["destination_tp"], case["page_layout"]) in missing]
    plan["protocol"]["iterations"] = 32
    plan["selection"] = {
        "source_plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "original_protocol": original_protocol, "selected_complete_curves": sorted(missing),
        "reason": "Qualified span does not cover8192. Fixed32-call averages retain every call and the10% CV gate; no outlier deletion or fastest-point selection.",
    }
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"supplement_required": True, "curves": sorted(missing), "cases": len(plan["cases"])}))


if __name__ == "__main__":
    main()
