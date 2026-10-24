"""Declare model-correlated DCP groups and their exact padded query geometry."""

import hashlib
import json
from pathlib import Path

import yaml


def get_ascend_dcp_attention_cases(model_path, case_root=None):
    model_path = Path(model_path)
    case_root = Path(case_root) if case_root else Path(__file__).resolve().parents[1] / "cases"
    config_file = model_path / "config.json"
    config = json.loads(config_file.read_text())
    model_file = case_root / "models" / f"{config['architectures'][0]}_cases.yaml"
    profiles = yaml.safe_load(model_file.read_text())["model_case_values"]["ascend_dcp_attention"]
    matching = [profile for profile in profiles if profile["checkpoint_quantization"] == "modelslim_w8a8"]
    if len(matching) != 1:
        raise ValueError("Expected exactly one declared DCP attention geometry")
    geometry = matching[0]
    for name in ("num_attention_heads", "kv_lora_rank", "qk_rope_head_dim", "index_topk"):
        if config.get(name) != geometry[name]:
            raise ValueError(f"Checkpoint differs from declared {name}")
    base_file = case_root / "base_ops/ascend_dcp_attention.yaml"
    base = yaml.safe_load(base_file.read_text())["common_case_values"]["ascend_dcp_attention"]
    cases = []
    for tensor in geometry["tensor_parallel_sizes"]:
        if tensor < 1 or geometry["num_attention_heads"] % tensor:
            raise ValueError("Attention TP must divide the head count")
        buckets = [value for value in base["graph_buckets"] if value % tensor == 0]
        if not buckets or max(buckets) < max(base["decode_batches"]):
            raise ValueError("Graph buckets do not cover the declared active batches")
        for dcp in geometry["dcp_sizes"]:
            if dcp > tensor or tensor % dcp:
                continue
            for remap in base["remap_topk"]:
                structural = {
                    "local_heads": geometry["num_attention_heads"] // tensor,
                    "dcp_size": dcp,
                    "remap_topk": remap,
                    "page_size": geometry["page_size"],
                    "sparse_topk": geometry["index_topk"],
                    "kv_lora_rank": geometry["kv_lora_rank"],
                    "rope_dim": geometry["qk_rope_head_dim"],
                    "kv_interleave_size": geometry["page_size"],
                }

                def add(phase, active, padded, query, prefix, length):
                    local_length = (length + dcp - 1) // dcp
                    pages = (local_length + geometry["page_size"] - 1) // geometry["page_size"]
                    cases.append(
                        {
                            **structural,
                            "phase": phase,
                            "execution_mode": "eager" if phase == "prefill" else "graph",
                            "active_batch": active,
                            "padded_batch": padded,
                            "query_tokens_per_request": query,
                            "tokens": active * query if phase == "prefill" else padded,
                            "sequence_length": length,
                            "prefix": prefix,
                            "pool_pages": max(
                                (geometry["reference_global_pool_pages"] + dcp - 1) // dcp, 1 + active * pages
                            ),
                            "block_table_width": pages
                            if phase == "prefill"
                            else (geometry["context_limit"] + geometry["page_size"] * dcp - 1)
                            // (geometry["page_size"] * dcp),
                        }
                    )

                for query in base["prefill_tokens"]:
                    for active in base["prefill_batches"]:
                        if query * active <= base["max_prefill_tokens"]:
                            for prefix in base["prefix_tokens"]:
                                add("prefill", active, active, query, prefix, query + prefix)
                for active in base["decode_batches"]:
                    padded = next(value for value in buckets if value >= active)
                    for length in base["context_lengths"]:
                        add("decode", active, padded, 1, 0, length)
    if not cases:
        raise ValueError("DCP declaration resolved to no cases")
    return {
        "schema": "aisimulate-ascend-dcp-attention-plan/v1",
        "status": "experimental_collection_plan",
        "model_path": str(model_path),
        "cases": cases,
        "protocol": {name: base[name] for name in ("iterations", "warmup", "repeats")},
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (config_file, model_file, base_file)
        },
        "boundary_contract": "sglang_npu_dcp_storage_attention_chain/v1",
        "scope": (
            "Actual NPU DCP cache writes, query gather, partial sparse attention and FP32 output merge; "
            "full-indexer layers also include indexer KV writes and index remapping. "
            "Excludes lightning indexer computation, projections and BMM. "
            "No additional cache-write operator may be added to this boundary."
        ),
        "source_contract": {
            "revision": "baa9befa381511cd4d56f83980951e5a45fffc6d",
            "sparse_index_ownership": (
                "hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py:509 "
                "passes forward_context.get_attn_backend().page_size as interleave_size"
            ),
            "kv_lengths_and_pages": (
                "hardware_backend/npu/attention/ascend_backend.py:469 "
                "uses get_dcp_lens(..., interleave_size=self.page_size) and page_stride=page_size*dcp_size"
            ),
            "output_merge": "layers/dcp/comm.py:126 uses FP32 all_to_all_single and npu_attention_update",
        },
    }
