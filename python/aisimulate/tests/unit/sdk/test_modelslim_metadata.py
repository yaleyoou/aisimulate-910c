import json

import pytest

from aisimulate_core.modelslim import read_glm_modelslim

pytestmark = pytest.mark.unit


def fixture(tmp_path):
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
    }
    values = {"model_quant_type": "W8A8", "model.embed_tokens.weight": "FLOAT", "lm_head.weight": "FLOAT"}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            values[prefix + "self_attn." + name + ".weight"] = "W8A8"
        values[prefix + "self_attn.kv_b_proj.weight"] = "FLOAT"
        groups = (
            [prefix + "mlp."]
            if layer == 0
            else [prefix + "mlp.shared_experts.", *(prefix + f"mlp.experts.{expert}." for expert in range(2))]
        )
        for group in groups:
            for name in ("gate_proj", "up_proj", "down_proj"):
                values[group + name + ".weight"] = "W8A8_DYNAMIC"
        if layer:
            values[prefix + "mlp.gate.weight"] = "FLOAT"
    path = tmp_path / "quant_model_description.json"
    path.write_text(json.dumps(values))
    return config, values, path


def test_compact_metadata_preserves_mixed_precision_and_expert_count(tmp_path):
    config, _, _ = fixture(tmp_path)
    result = read_glm_modelslim(str(tmp_path), config)
    assert result["primary_gemm_mode"] == "w8a8"
    assert result["moe_mode"] == "w8a8_dynamic"
    assert result["layers"][0]["attention"]["kv_b_proj"] == "FLOAT"
    assert result["layers"][0]["mlp"]["kind"] == "dense"
    assert result["layers"][1]["mlp"]["experts"] == 2
    assert result["layers"][1]["mlp"]["routed"]["down_proj"] == "W8A8_DYNAMIC"
    assert result["lm_head"] == "FLOAT"


def test_missing_or_conflicting_fused_projection_never_falls_back(tmp_path):
    config, values, path = fixture(tmp_path)
    del values["model.layers.0.self_attn.q_b_proj.weight"]
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="Missing or unsupported"):
        read_glm_modelslim(str(tmp_path), config)
    config, values, path = fixture(tmp_path)
    values["model.layers.0.mlp.up_proj.weight"] = "FLOAT"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="Fused gate/up"):
        read_glm_modelslim(str(tmp_path), config)


def test_mixed_expert_precision_requires_an_explicit_representation(tmp_path):
    config, values, path = fixture(tmp_path)
    values["model.layers.1.mlp.experts.1.down_proj.weight"] = "FLOAT"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="Per-expert mixed"):
        read_glm_modelslim(str(tmp_path), config)
