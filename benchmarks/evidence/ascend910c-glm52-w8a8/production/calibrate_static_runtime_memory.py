"""Derive the pre-KV runtime reservation from instrumented United snapshots."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh calibration report")
    records, sources = [], {}
    for campaign in args.campaigns:
        launch = json.loads((campaign / "launch.json").read_text())
        if launch["checkpoint_sha256"] != "41795173ae457a64056e1c05be9bbbf889f935431f4a42fa4f4e4c6a32aa931c":
            raise ValueError("Memory observations must use the current checkpoint")
        command = launch["command"]
        for name, value in (("--tp-size", "16"), ("--dp-size", "2"), ("--ep-size", "16"), ("--mem-fraction-static", "0.90")):
            if command[command.index(name) + 1] != value:
                raise ValueError("Unqualified memory deployment")
        paths = sorted((campaign / "memory").glob("rank-*.jsonl"))
        if len(paths) != 16:
            raise ValueError("Incomplete worker memory inventory")
        ranks = set()
        for path in paths:
            payload = path.read_bytes()
            snapshots = {row["boundary"]: row for row in map(json.loads, payload.splitlines())}
            loaded = snapshots["load_model:after"]
            before = snapshots["alloc_memory_pool:before"]
            allocated = snapshots["alloc_memory_pool:after"]
            graph = snapshots["init_cuda_graphs:after"]
            rank = loaded["rank"]
            if rank in ranks or any(row["rank"] != rank for row in snapshots.values()):
                raise ValueError("Duplicate or mismatched memory rank")
            ranks.add(rank)
            weights = loaded["model_inventory"]["unique_storage_bytes"]
            if weights != 52_471_997_948:
                raise ValueError("Resident parameter inventory differs from the audited layout")
            reserve = before["total_bytes"] - before["free_bytes"] - weights
            if reserve <= 0 or before["total_bytes"] < 65_787_658_240:
                raise ValueError("Invalid observed static memory reservation")
            records.append({
                "campaign": campaign.name, "rank": rank, "model_storage_bytes": weights,
                "reported_device_capacity_bytes": before["total_bytes"],
                "static_other_bytes": reserve,
                "usable_kv_tokens": allocated["max_total_num_tokens"],
                "after_graph_free_bytes": graph["free_bytes"],
                "graph_free_consumed_bytes": allocated["free_bytes"] - graph["free_bytes"],
            })
            sources[str(path)] = hashlib.sha256(payload).hexdigest()
        if ranks != set(range(16)):
            raise ValueError("Incomplete rank placement")
    report = {
        "schema": "aisimulate-ascend-static-runtime-memory/v1",
        "status": "measured",
        "scope": "Pinned SGLang0.5.22/CANN9.1.0.B243/driver26.1.1, current GLM ModelSlim, United TP16/ADP2/EP16",
        "method": "maximum(total_device_bytes - free_bytes_before_KV_allocation - unique_model_storage_bytes)",
        "static_other_bytes": max(row["static_other_bytes"] for row in records),
        "records": records, "sources": sources,
        "limitations": [
            "Includes allocator slack and runtime allocations before KV allocation; not a datasheet hardware constant.",
            "SGLang reserves graph/execution headroom outside its static-memory fraction.",
            "The SDK applies its existing20% SGLang other-memory margin; prediction is conservative.",
            "Other parallel layouts have not received an independent resident-memory calibration.",
            "NCCL is not used; its compatibility memory bucket is zero, not an HCCL proxy.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"static_other_bytes": report["static_other_bytes"], "observations": len(records)}))


if __name__ == "__main__":
    main()
