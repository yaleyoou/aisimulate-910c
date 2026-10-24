"""Resolve the actual EP1 Ascend tensor-MoE shape and timing domain."""

import hashlib
import json
from pathlib import Path

import yaml


def get_ascend_moe_tensor_cases(model_path, case_root=None):
    model_path = Path(model_path)
    case_root = Path(case_root) if case_root else Path(__file__).resolve().parents[1] / "cases"
    config_path = model_path / "config.json"
    quant_path = model_path / "quant_model_description.json"
    config, quantization = json.loads(config_path.read_text()), json.loads(quant_path.read_text())
    if quantization.get("model.layers.3.mlp.experts.0.gate_proj.weight") != "W8A8_DYNAMIC":
        raise ValueError("Expected the ModelSlim dynamic W8A8 expert checkpoint")
    model_file = case_root / "models" / f"{config['architectures'][0]}_cases.yaml"
    profiles = yaml.safe_load(model_file.read_text())["model_case_values"]["ascend_moe_tensor"]
    matching = [profile for profile in profiles if profile["checkpoint_quantization"] == "modelslim_w8a8"]
    if len(matching) != 1:
        raise ValueError("Expected exactly one declared tensor-MoE geometry")
    geometry = matching[0]
    for field in ("hidden_size", "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok"):
        if config.get(field) != geometry[field]:
            raise ValueError(f"Checkpoint differs from declared {field}")
    base_file = case_root / "base_ops/ascend_moe_tensor.yaml"
    base = yaml.safe_load(base_file.read_text())["common_case_values"]["ascend_moe_tensor"]
    cases = []
    for parallel in geometry["tensor_parallel_sizes"]:
        if parallel < 1 or geometry["moe_intermediate_size"] % parallel:
            raise ValueError("Tensor partition must divide the expert intermediate dimension")
        for mode in base["execution_modes"]:
            for tokens in base["tokens"]:
                cases.append(
                    {
                        "hidden_size": geometry["hidden_size"],
                        "num_experts": geometry["n_routed_experts"],
                        "top_k": geometry["num_experts_per_tok"],
                        "intermediate_size": geometry["moe_intermediate_size"] // parallel,
                        "distribution": "balanced",
                        "execution_mode": mode,
                        "tokens": tokens,
                    }
                )
    return {
        "schema": "aisimulate-ascend-moe-tensor-plan/v1",
        "status": "experimental_collection_plan",
        "model_path": str(model_path),
        "cases": cases,
        "protocol": {name: base[name] for name in ("weight_copies", "iterations", "warmup", "repeats")},
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (config_path, quant_path, model_file, base_file)
        },
        "scope": "EP1 AscendTPDispatcher with balanced distinct top8; excludes router, shared expert and all-reduce.",
    }
