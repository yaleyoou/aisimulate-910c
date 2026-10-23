"""Declare DeepEP routing, placement and actual serving execution branches."""

import hashlib
import json
from pathlib import Path

import yaml


def get_ascend_deepep_cases(model_path, case_root=None):
    model_path = Path(model_path)
    case_root = Path(case_root) if case_root else Path(__file__).resolve().parents[1] / "cases"
    config_file = model_path / "config.json"
    quant_file = model_path / "quant_model_description.json"
    config = json.loads(config_file.read_text())
    if json.loads(quant_file.read_text()).get("model.layers.3.mlp.experts.0.gate_proj.weight") != "W8A8_DYNAMIC":
        raise ValueError("Expected ModelSlim dynamic W8A8 experts")
    model_file = case_root / "models" / f"{config['architectures'][0]}_cases.yaml"
    geometry = yaml.safe_load(model_file.read_text())["model_case_values"]["ascend_deepep"]
    if len(geometry) != 1 or geometry[0]["checkpoint_quantization"] != "modelslim_w8a8":
        raise ValueError("Expected one declared Ascend DeepEP geometry")
    geometry = geometry[0]
    for name in ("hidden_size", "n_routed_experts", "num_experts_per_tok"):
        if config.get(name) != geometry[name]:
            raise ValueError(f"Model differs from declared {name}")
    base_file = case_root / "base_ops/ascend_deepep.yaml"
    base = yaml.safe_load(base_file.read_text())["common_case_values"]["ascend_deepep"]
    cases = []
    for world, local in geometry["placements"]:
        if world % local or geometry["n_routed_experts"] % world:
            raise ValueError("Invalid DeepEP rank/expert placement")
        for phase in ("normal", "low_latency"):
            for distribution in base["distributions"]:
                for tokens in base[phase + "_tokens"]:
                    if phase == "low_latency" and tokens > geometry["max_dispatch_tokens_per_rank"]:
                        raise ValueError("Low latency input exceeds the declared dispatch limit")
                    for mode in base[phase + "_execution_modes"]:
                        cases.append(
                            {
                                "phase": phase,
                                "num_ranks": world,
                                "ranks_per_node": local,
                                "rank_layout": "contiguous",
                                "tokens_per_rank": tokens,
                                "distribution": distribution,
                                "execution_mode": mode,
                                "hidden_size": geometry["hidden_size"],
                                "num_experts": geometry["n_routed_experts"],
                                "topk": geometry["num_experts_per_tok"],
                                "max_dispatch_tokens_per_rank": geometry["max_dispatch_tokens_per_rank"],
                            }
                        )
    return {
        "schema": "aisimulate-ascend-deepep-plan/v1",
        "status": "experimental_collection_plan",
        "cases": cases,
        "protocol": {name: base[name] for name in ("iterations", "warmup", "repeats")},
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (config_file, quant_file, model_file, base_file)
        },
        "scope": (
            "Pinned SGLang dispatcher implementations, async normal and receive-hook low latency; "
            "INT8 token quantization. Synthetic full-payload routing controls; "
            "not measured checkpoint routing or serving overlap."
        ),
    }
