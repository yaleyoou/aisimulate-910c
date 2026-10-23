"""Declare count vectors separately from grouped-matmul storage capacity."""

import hashlib
import json
from pathlib import Path

import yaml


def get_ascend_moe_chain_cases(model_path, case_root=None):
    model_path = Path(model_path)
    case_root = Path(case_root) if case_root else Path(__file__).resolve().parents[1] / "cases"
    config_file, quant_file = model_path / "config.json", model_path / "quant_model_description.json"
    config, quantization = json.loads(config_file.read_text()), json.loads(quant_file.read_text())
    if quantization.get("model.layers.3.mlp.experts.0.gate_proj.weight") != "W8A8_DYNAMIC":
        raise ValueError("Expected dynamic W8A8 routed experts")
    model_file = case_root / "models" / f"{config['architectures'][0]}_cases.yaml"
    profiles = yaml.safe_load(model_file.read_text())["model_case_values"]["ascend_moe_chain"]
    matching = [value for value in profiles if value["checkpoint_quantization"] == "modelslim_w8a8"]
    if len(matching) != 1:
        raise ValueError("Expected one declared MoE geometry")
    geometry = matching[0]
    for name in ("hidden_size", "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok"):
        if geometry[name] != config.get(name):
            raise ValueError(f"Model differs from declared {name}")
    base_file = case_root / "base_ops/ascend_moe_chain.yaml"
    base = yaml.safe_load(base_file.read_text())["common_case_values"]["ascend_moe_chain"]
    cases = {}
    recipes = 0
    for parallel in geometry["expert_parallel_sizes"]:
        experts = geometry["n_routed_experts"] // parallel
        if experts * parallel != geometry["n_routed_experts"]:
            raise ValueError("Expert partition is not integral")
        capacity = parallel * base["decode_capacity_per_source"] * geometry["num_experts_per_tok"]
        for phase in ("prefill", "decode"):
            for distribution in base["distributions"]:
                for tokens in base["received_tokens"]:
                    if phase == "decode" and tokens > capacity:
                        continue
                    if distribution == "balanced":
                        counts = [tokens // experts + int(index < tokens % experts) for index in range(experts)]
                    elif distribution == "single_expert":
                        counts = [tokens] + [0] * (experts - 1)
                    else:
                        raise ValueError("Unknown expert distribution")
                    for mode in base["execution_modes"]:
                        recipes += 1
                        case = {
                            "phase": phase,
                            "local_experts": experts,
                            "hidden_size": geometry["hidden_size"],
                            "intermediate_size": geometry["moe_intermediate_size"],
                            "received_tokens": tokens,
                            "physical_rows": tokens if phase == "prefill" else capacity,
                            "expert_counts": counts,
                            "execution_mode": mode,
                            "group_list_type": 1,
                        }
                        identity = json.dumps(case, sort_keys=True, separators=(",", ":"))
                        cases.setdefault(identity, {**case, "distributions": []})["distributions"].append(distribution)
    return {
        "schema": "aisimulate-ascend-moe-chain-plan/v1",
        "status": "experimental_collection_plan",
        "model_path": str(model_path),
        "cases": list(cases.values()),
        "recipe_invocations": recipes,
        "unique_invocations": len(cases),
        "protocol": {name: base[name] for name in ("weight_copies", "iterations", "warmup", "repeats")},
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (config_file, quant_file, model_file, base_file)
        },
        "scope": (
            "Synthetic balanced/single-expert count profiles; exact 6144/2048/INT8NZ/BF16 geometry. "
            "EP16 decode capacity4096 observed; EP32 capacity8192 is a declared standalone buffer case "
            "until distributed dispatch verifies it. Not real checkpoint routing distributions."
        ),
    }
