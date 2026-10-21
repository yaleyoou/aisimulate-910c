import argparse
import collections
import csv
import datetime
import hashlib
import json
import pathlib
import re
import struct
import subprocess


def digest_bytes(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def inspect(model_dir, output_dir, repository):
    config = json.loads((model_dir / "config.json").read_text())
    quant = json.loads((model_dir / "quant_model_description.json").read_text())
    index = json.loads((model_dir / "quant_model_weights.safetensors.index.json").read_text())
    weights = index["weight_map"]
    metadata = {}
    for name in (
        "config.json", "quant_model_description.json", "quant_model_weights.safetensors.index.json",
        "generation_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
        "README.md", "GLM-5.1_best_practice.yaml",
    ):
        path = model_dir / name
        if path.is_file():
            content = path.read_bytes()
            metadata[name] = {"path": str(path), "size_bytes": len(content), "sha256": digest_bytes(content)}
    shards = []
    tensors = {}
    issues = []
    for name in sorted(set(weights.values())):
        path = model_dir / name
        if not path.is_file():
            issues.append({"missing_shard": name})
            continue
        before = path.stat()
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            header_length = struct.unpack("<Q", length_bytes)[0]
            if header_length > 128 * 1024 * 1024:
                raise ValueError("Unexpectedly large tensor header: " + str(path))
            header_bytes = stream.read(header_length)
            header = json.loads(header_bytes)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            issues.append({"changed_during_inspection": name})
        count = 0
        for tensor_name, tensor in header.items():
            if tensor_name == "__metadata__":
                continue
            count += 1
            if tensor_name in tensors:
                issues.append({"duplicate_tensor": tensor_name})
            tensors[tensor_name] = {**tensor, "shard": name}
        shards.append({
            "path": str(path), "name": name, "size_bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns, "header_sha256": digest_bytes(header_bytes),
            "header_length": header_length, "tensor_count": count,
            "payload_sha256": None,
        })
    for name, shard in weights.items():
        if name not in tensors or tensors[name]["shard"] != shard:
            issues.append({"index_header_mismatch": name, "index_shard": shard})
    for name in tensors.keys() - weights.keys():
        issues.append({"unindexed_tensor": name})
    groups = {}
    dtype_counts = collections.Counter()
    dtype_bytes = collections.Counter()
    weight_counts = collections.Counter()
    for name, tensor in tensors.items():
        tensor_bytes = tensor["data_offsets"][1] - tensor["data_offsets"][0]
        dtype_counts[tensor["dtype"]] += 1
        dtype_bytes[tensor["dtype"]] += tensor_bytes
        if not name.endswith(".weight"):
            continue
        match = re.search(r"model\.layers\.(\d+)\.", name)
        layer = int(match.group(1)) if match else None
        if layer is None:
            phase = "outside_layers"
        elif layer >= config["num_hidden_layers"]:
            phase = "mtp"
        elif layer < config["first_k_dense_replace"]:
            phase = "dense_prefix"
        else:
            phase = "moe_backbone"
        module = re.sub(r"model\.layers\.\d+\.", "", name)
        module = re.sub(r"\.experts\.\d+\.", ".experts.*.", module)
        mode = quant.get(name, "NOT_DECLARED")
        weight_counts[mode] += 1
        key = (phase, module, mode, tensor["dtype"])
        group = groups.setdefault(key, {"count": 0, "bytes": 0, "shapes": set(), "layers": set(), "sample": name})
        group["count"] += 1
        group["bytes"] += tensor_bytes
        group["shapes"].add(tuple(tensor["shape"]))
        if layer is not None:
            group["layers"].add(layer)
        if mode in {"W8A8", "W8A8_DYNAMIC"} and tensor["dtype"] != "I8":
            issues.append({"quant_dtype_mismatch": name, "quant": mode, "dtype": tensor["dtype"]})
    with (output_dir / "quantization-map.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer_group", "module", "quantization", "stored_dtype", "tensor_count", "stored_bytes", "shapes", "layer_ids", "sample"])
        for key, group in sorted(groups.items()):
            writer.writerow([*key, group["count"], group["bytes"], json.dumps(sorted(group["shapes"])), json.dumps(sorted(group["layers"])), group["sample"]])
    fields = (
        "architectures", "model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
        "first_k_dense_replace", "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
        "moe_intermediate_size", "num_attention_heads", "num_key_value_heads", "kv_lora_rank",
        "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "head_dim",
        "index_head_dim", "index_n_heads", "index_topk", "index_topk_freq", "index_skip_topk_offset",
        "num_nextn_predict_layers", "dtype", "moe_router_dtype", "max_position_embeddings", "vocab_size",
    )
    total_bytes = sum(dtype_bytes.values())
    expected_bytes = index.get("metadata", {}).get("total_size")
    metadata_size_discrepancy = None
    if expected_bytes is not None and total_bytes != expected_bytes:
        rotation = tensors.get("rot.weight")
        rotation_bytes = rotation["data_offsets"][1] - rotation["data_offsets"][0] if rotation else None
        metadata_size_discrepancy = {
            "index_total_size": expected_bytes, "header_tensor_bytes": total_bytes,
            "difference_bytes": total_bytes - expected_bytes,
            "rotation_tensor_bytes": rotation_bytes,
            "difference_equals_rotation_tensor": total_bytes - expected_bytes == rotation_bytes,
            "resolution": "Use header byte totals for storage accounting; writer intent and full payload integrity unverified.",
        }
    report = {
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_path": str(model_dir),
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "model_fields": {name: config.get(name) for name in fields},
        "indexer_types": dict(collections.Counter(config["indexer_types"])),
        "mlp_layer_types": dict(collections.Counter(config["mlp_layer_types"])),
        "metadata_sources": metadata,
        "quantized_weight_counts": dict(weight_counts),
        "all_tensor_dtype_counts": dict(dtype_counts),
        "all_tensor_dtype_bytes": dict(dtype_bytes),
        "total_tensor_payload_bytes": total_bytes,
        "total_tensor_payload_gib": total_bytes / 1024 ** 3,
        "total_shard_file_bytes": sum(shard["size_bytes"] for shard in shards),
        "shard_count": len(shards), "indexed_tensor_count": len(weights),
        "header_tensor_count": len(tensors), "integrity_issues": issues,
        "metadata_size_discrepancy": metadata_size_discrepancy,
        "rotation_metadata": {name: quant.get(name) for name in ("is_rot_used", "optional", "group_size")},
        "identity_limits": [
            "Metadata and tensor headers hashed; full tensor payloads were not hashed.",
            "This manifest is not an immutable checkpoint revision or verification of historic measurements.",
            "Stored dtypes are not proof of runtime compute dtypes or allocation layouts.",
            "No model inference or device allocation was performed.",
        ],
    }
    write_json(output_dir / "model-inventory.json", report)
    write_json(output_dir / "weight-shards.json", shards)
    print(json.dumps({"shards": len(shards), "tensors": len(tensors), "gib": report["total_tensor_payload_gib"], "issues": len(issues), "quant_groups": len(groups)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--repository", type=pathlib.Path, default=pathlib.Path.cwd())
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inspect(args.model, args.output, args.repository)
