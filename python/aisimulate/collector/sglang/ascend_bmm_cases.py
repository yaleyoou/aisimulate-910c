"""Expand declared MLA BMM shapes without changing production stride contracts."""

import hashlib
import json
from pathlib import Path

import yaml


def get_ascend_bmm_cases(model_path, case_root=None, device_memory_bytes=None):
    model_path = Path(model_path)
    case_root = Path(case_root) if case_root else Path(__file__).resolve().parents[1] / "cases"
    config_file = model_path / "config.json"
    config = json.loads(config_file.read_text())
    model_file = case_root / "models" / f"{config['architectures'][0]}_cases.yaml"
    profiles = yaml.safe_load(model_file.read_text())["model_case_values"]["ascend_bmm"]
    matching = [value for value in profiles if value["checkpoint_quantization"] == "modelslim_w8a8"]
    if len(matching) != 1:
        raise ValueError("Expected one declared NPU MLA BMM geometry")
    geometry = matching[0]
    for name in ("num_attention_heads", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
        if config.get(name) != geometry[name]:
            raise ValueError(f"Model differs from declared {name}")
    base_file = case_root / "base_ops/ascend_bmm.yaml"
    base = yaml.safe_load(base_file.read_text())["common_case_values"]["ascend_bmm"]
    cases = []
    for parallel in geometry["tensor_parallel_sizes"]:
        if geometry["num_attention_heads"] % parallel:
            raise ValueError("Attention heads must divide TP")
        heads = geometry["num_attention_heads"] // parallel
        for role in base["roles"]:
            query = role == "query_nope"
            if role not in {"query_nope", "output_prefill", "output_decode"}:
                raise ValueError("Unknown BMM role")
            inner = geometry["qk_nope_head_dim"] if query else geometry["kv_lora_rank"]
            width = geometry["kv_lora_rank"] if query else geometry["v_head_dim"]
            physical_inner = inner + geometry["qk_rope_head_dim"] if query else inner
            kernel = (
                "torch.ops.npu.batch_matmul_transpose"
                if role == "output_decode"
                else "torch_npu.npu_transpose_batchmatmul"
            )
            for mode in base["execution_modes"]:
                for tokens in base["decode_tokens"] if role == "output_decode" else base["tokens"]:
                    cases.append(
                        {
                            "role": role,
                            "local_heads": heads,
                            "tokens": tokens,
                            "input_width": inner,
                            "output_width": width,
                            "input_stride": [heads * physical_inner, physical_inner, 1],
                            "weight_stride": [inner * width, 1, inner] if query else [inner * width, width, 1],
                            "execution_mode": mode,
                            "kernel_source": kernel,
                            "dtype": "bfloat16",
                            "format": 2,
                        }
                    )
    memory_exclusions = []
    if device_memory_bytes is not None:
        accepted = []
        for case in cases:
            retained_outputs = base["iterations"] * case["tokens"] * case["local_heads"] * case["output_width"] * 2
            weights = base["weight_copies"] * case["local_heads"] * case["input_width"] * case["output_width"] * 2
            inputs = case["tokens"] * case["input_stride"][0] * 2
            estimated_bytes = retained_outputs + weights + inputs
            if estimated_bytes > device_memory_bytes * 0.75:
                memory_exclusions.append({"case": case, "estimated_bytes": estimated_bytes})
            else:
                accepted.append(case)
        cases = accepted
    return {
        "schema": "aisimulate-ascend-bmm-plan/v1",
        "status": "experimental_collection_plan",
        "model_path": str(model_path),
        "cases": cases,
        "memory_feasibility": {
            "device_memory_bytes": device_memory_bytes,
            "fraction": 0.75,
            "excluded": memory_exclusions,
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (config_file, model_file, base_file)
        },
        "protocol": {name: base[name] for name in ("weight_copies", "iterations", "warmup", "repeats")},
        "source_contract": (
            "SGLang baa9befa381511cd4d56f83980951e5a45fffc6d "
            "deepseek_v2_attention_mla_npu.py:466-474,574-595; "
            "observed rank0/8 input strides in operator-input-state-node31-20261022-attempt2."
            " Decode batch axis is bounded by United's 32 requests per attention-DP rank;"
            " the prior overbroad grid and custom-kernel batchIdx failures remain in attempt2."
        ),
    }
