"""Generate replica-copy cases for the pinned Ascend DSA KV layout."""

import hashlib
import json
from pathlib import Path

import yaml


def get_ascend_pd_cases(model_path, case_root=None):
    model_path = Path(model_path)
    case_root = Path(case_root) if case_root else Path(__file__).resolve().parents[1] / "cases"
    config_path = model_path / "config.json"
    quant_path = model_path / "quant_model_description.json"
    config = json.loads(config_path.read_text())
    if json.loads(quant_path.read_text()).get("model_quant_type") != "W8A8":
        raise ValueError("Expected the declared ModelSlim checkpoint")
    model_file = case_root / "models" / f"{config['architectures'][0]}_cases.yaml"
    profiles = yaml.safe_load(model_file.read_text())["model_case_values"]["ascend_pd"]
    if len(profiles) != 1:
        raise ValueError("Expected one declared PD memory layout")
    profile = profiles[0]
    for name in ("num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim", "index_head_dim"):
        if config.get(name) != profile[name]:
            raise ValueError(f"Model differs from declared {name}")
    base_file = case_root / "base_ops/ascend_pd.yaml"
    base = yaml.safe_load(base_file.read_text())["common_case_values"]["ascend_pd"]
    cases = []
    for source_tp in profile["source_tensor_parallel_sizes"]:
        for destination_tp in profile["destination_tensor_parallel_sizes"]:
            if profile["ranks_per_node"] % source_tp or profile["ranks_per_node"] % destination_tp:
                raise ValueError("Attention groups must fit the declared node")
            for layout in base["page_layouts"]:
                for tokens in base["token_counts"]:
                    if tokens % profile["page_size"] or 2 * tokens > profile["reference_pool_tokens"]:
                        raise ValueError("Transfer and guard pages exceed the reference pool")
                    cases.append(
                        {
                            "source_tp": source_tp,
                            "destination_tp": destination_tp,
                            "tokens": tokens,
                            "page_layout": layout,
                            "protocol": "sdma",
                            "active_requests": 1,
                            "ranks_per_node": profile["ranks_per_node"],
                            "num_layers": profile["num_hidden_layers"],
                            "page_size": profile["page_size"],
                            "reference_pool_tokens": profile["reference_pool_tokens"],
                            "component_widths": [
                                profile["kv_lora_rank"],
                                profile["qk_rope_head_dim"],
                                profile["index_head_dim"],
                            ],
                            "dtype": profile["kv_dtype"],
                        }
                    )
    return {
        "schema": "aisimulate-ascend-pd-plan/v1",
        "cases": cases,
        "protocol": {name: base[name] for name in ("iterations", "warmup", "repeats")},
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (config_path, quant_path, model_file, base_file)
        },
        "scope": (
            "One request and actual MLA rank mapping; all receiver replicas move concurrently. "
            "Bootstrap/admission and concurrent-request contention excluded."
        ),
    }
