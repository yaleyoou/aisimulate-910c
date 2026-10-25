"""Freeze independent short/long-context and ragged-batch attention probes."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    base = json.loads(args.base_plan.read_text())
    cases = []
    for operation, heads in [
        ("lightning_indexer", 32),
        ("sparse_attention", 4),
        ("sparse_attention", 8),
        ("sparse_attention", 16),
        ("sparse_attention", 32),
        ("sparse_attention", 64),
    ]:
        for phase in ("prefill", "decode"):
            template = next(
                case
                for case in base["cases"]
                if case["operation"] == operation
                and case["local_heads"] == heads
                and case["phase"] == phase
                and case["execution_mode"] == ("eager" if phase == "prefill" else "graph")
                and case["active_batch"] == 1
            )
            shapes = (
                [(1, 4096, prefix) for prefix in (12288, 32768, 65536)] + [(3, 2731, 0), (4, 3072, 0)]
                if phase == "prefill"
                else [(1, length, 0) for length in (1, 65, 8192, 32768, 65536, 131072)]
            )
            if args.confirmation:
                shapes = (
                    [(1, 4096, prefix) for prefix in (49152, 98304)] + [(3, 3000, 0), (5, 1700, 0)]
                    if phase == "prefill"
                    else [(1, length, 0) for length in (8192, 49152, 98304)]
                )
            for batch, length, prefix in shapes:
                case = copy.deepcopy(template)
                sequence = length + prefix
                case.update(
                    active_batch=batch,
                    padded_batch=batch if phase == "prefill" else template["padded_batch"],
                    tokens=batch * length if phase == "prefill" else template["padded_batch"],
                    sequence_length=sequence,
                    prefix=prefix,
                    pool_pages=max(417, 1 + batch * ((sequence + 127) // 128)),
                    block_table_width=(sequence + 127) // 128 if phase == "prefill" else 1407,
                )
                cases.append(case)
    plan = {
        **base,
        "cases": cases,
        "purpose": "Independent out-of-grid validation; original calibration curves remain unchanged",
        "base_plan_sha256": hashlib.sha256(args.base_plan.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "plan": str(args.output)}))


if __name__ == "__main__":
    main()
