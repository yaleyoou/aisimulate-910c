import json
from pathlib import Path


def write_glm52_modelslim(directory):
    source = Path(__file__).resolve().parents[3] / "src/aisimulate_core/model_configs/zai-org--GLM-5.2_config.json"
    config = json.loads(source.read_text())
    config.pop("quantization_config", None)
    config.pop("quant_algo", None)
    (directory / "config.json").write_text(json.dumps(config))
    quant = {"model_quant_type": "W8A8", "model.embed_tokens.weight": "FLOAT", "lm_head.weight": "FLOAT"}
    for layer in range(78):
        prefix = f"model.layers.{layer}."
        for projection in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            quant[prefix + "self_attn." + projection + ".weight"] = "W8A8"
        quant[prefix + "self_attn.kv_b_proj.weight"] = "FLOAT"
        if layer < 3 or (layer - 2) % 4 == 0:
            for projection, dtype in (("wq_b", "W8A8"), ("wk", "FLOAT"), ("weights_proj", "FLOAT")):
                quant[prefix + "self_attn.indexer." + projection + ".weight"] = dtype
        if layer < 3:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                quant[prefix + "mlp." + projection + ".weight"] = "W8A8_DYNAMIC"
        else:
            quant[prefix + "mlp.gate.weight"] = "FLOAT"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                quant[prefix + "mlp.shared_experts." + projection + ".weight"] = "W8A8_DYNAMIC"
                for expert in range(256):
                    quant[prefix + f"mlp.experts.{expert}." + projection + ".weight"] = "W8A8_DYNAMIC"
    (directory / "quant_model_description.json").write_text(json.dumps(quant))
    return str(directory)
