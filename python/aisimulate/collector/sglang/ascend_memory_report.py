"""Reconcile complete Ascend startup observations without inventing reservations."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

BOUNDARIES = (
    "load_model:before",
    "load_model:after",
    "alloc_memory_pool:before",
    "alloc_memory_pool:after",
    "init_cuda_graphs:before",
    "init_cuda_graphs:after",
)


def summarize(directory, observer_source, ranks=16):
    paths = list(directory.glob("rank-*.jsonl.gz")) or list(directory.glob("rank-*.jsonl"))
    files = {int(path.name.split(".")[0].split("-")[1]): path for path in paths}
    if set(files) != set(range(ranks)):
        raise ValueError("Memory observations must cover every rank")
    result = {
        "schema": "aisimulate-ascend-worker-memory-report/v1",
        "status": "validated_observations",
        "production_memory_model_ready": False,
        "sources": {},
        "ranks": [],
        "limits": [
            "Allocator reservations, free-memory changes and storage extents have different ownership/lifetimes.",
            "Model-reachable storage includes parameters, buffers and direct tensor attributes, not only weights.",
            "Do not add independently measured HCCL/graph buffers to a total that already includes them.",
            "One deployment's initialized capacity does not establish other parallelism or PD roles.",
        ],
    }
    identities = set()
    for rank, path in sorted(files.items()):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            records = [json.loads(line) for line in stream]
        if tuple(record["boundary"] for record in records) != BOUNDARIES:
            raise ValueError(f"Rank{rank} has incomplete or repeated startup phases")
        for record in records:
            if record["schema"] != "aisimulate-ascend-worker-memory/v1" or record["rank"] != rank:
                raise ValueError("Memory observation identity mismatch")
            for field in ("allocated_bytes", "reserved_bytes", "free_bytes", "total_bytes"):
                value = record[field]
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid {field}")
            if record["allocated_bytes"] > record["reserved_bytes"] or record["free_bytes"] > record["total_bytes"]:
                raise ValueError("Impossible allocator/device accounting")
            identities.add((record["sglang_commit"], record["observer_sha256"]))
        if records != sorted(records, key=lambda record: record["time_ns"]):
            raise ValueError("Startup boundaries are not time ordered")
        inventories = {}
        for record in (records[1], records[5]):
            inventory = record["model_inventory"]
            storages = {}
            for tensor in inventory["tensors"]:
                if not tensor["resident_npu"]:
                    continue
                pointer, size = tensor["storage_pointer"], tensor["storage_bytes"]
                if type(pointer) is not int or pointer < 0 or type(size) is not int or size < 0:
                    raise ValueError("Invalid NPU storage extent")
                if pointer in storages and storages[pointer] != size:
                    raise ValueError("Aliased tensors disagree about their storage extent")
                storages[pointer] = size
            if (
                sum(storages.values()) != inventory["unique_storage_bytes"]
                or len(storages) != inventory["unique_storages"]
            ):
                raise ValueError("Model inventory totals are inconsistent")
            if inventory["unique_storage_bytes"] > record["allocated_bytes"]:
                raise ValueError("Model storage exceeds the observed tensor allocation")
            inventories[record["boundary"]] = {
                "unique_storage_bytes": inventory["unique_storage_bytes"],
                "tensor_count": len(inventory["tensors"]),
                "unique_storages": inventory["unique_storages"],
            }
        after_pool = records[3]
        tokens = after_pool["max_total_num_tokens"]
        if type(tokens) is not int or tokens <= 0 or tokens % 128:
            raise ValueError("Unexpected capacity for the audited 128-token paged layout")
        kv_bytes = 0
        for name, dimension in (("k_buffer", 512), ("v_buffer", 64), ("index_k_buffer", 128)):
            tensor = after_pool["kv_pool_tensors"][name]
            expected_shape = [78, tokens // 128 + 1, 128, 1, dimension]
            expected_bytes = 78 * (tokens + 128) * dimension * 2
            if (
                tensor["shape"] != expected_shape
                or tensor["dtype"] != "torch.bfloat16"
                or tensor["bytes"] != expected_bytes
            ):
                raise ValueError(f"Unexpected GLM BF16 KV allocation in {name}")
            kv_bytes += tensor["bytes"]
        if kv_bytes > after_pool["allocated_bytes"]:
            raise ValueError("KV storage exceeds the observed tensor allocation")
        deltas = []
        for before, after in zip(records[::2], records[1::2], strict=True):
            reserved_delta = after["reserved_bytes"] - before["reserved_bytes"]
            free_consumed = before["free_bytes"] - after["free_bytes"]
            deltas.append(
                {
                    "phase": before["boundary"].split(":")[0],
                    "allocated_delta_bytes": after["allocated_bytes"] - before["allocated_bytes"],
                    "reserved_delta_bytes": reserved_delta,
                    "free_consumed_bytes": free_consumed,
                    "free_consumed_minus_reserved_delta_bytes": free_consumed - reserved_delta,
                }
            )
        result["sources"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["ranks"].append(
            {
                "rank": rank,
                "device": after_pool["device"],
                "usable_tokens": tokens,
                "kv_storage_bytes": kv_bytes,
                "kv_bytes_per_token": 109824,
                "sentinel_bytes": 14057472,
                "after_graph_free_bytes": records[5]["free_bytes"],
                "model_reachable_storage": inventories,
                "phase_deltas": deltas,
            }
        )
    if len(identities) != 1:
        raise ValueError("Ranks used different runtime/observer sources")
    result["sglang_commit"], result["observer_sha256"] = next(iter(identities))
    if (
        result["sglang_commit"] != "baa9befa381511cd4d56f83980951e5a45fffc6d"
        or result["observer_sha256"] != hashlib.sha256(observer_source.read_bytes()).hexdigest()
    ):
        raise ValueError("Runtime/observer source does not match the audited campaign")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-source", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory, args.observer_source)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "ranks": len(result["ranks"])}))


if __name__ == "__main__":
    main()
