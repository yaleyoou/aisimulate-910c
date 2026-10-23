import json
from pathlib import Path

import pytest

from collector.sglang.ascend_moe_chain_cases import get_ascend_moe_chain_cases

pytestmark = pytest.mark.unit


def test_count_vectors_deduplicate_profiles_but_preserve_physical_capacity(tmp_path):
    config = Path(__file__).resolve().parents[3] / "src/aisimulate_core/model_configs/zai-org--GLM-5.2_config.json"
    (tmp_path / "config.json").write_bytes(config.read_bytes())
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps({"model.layers.3.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC"})
    )
    plan = get_ascend_moe_chain_cases(tmp_path)
    assert plan["recipe_invocations"] == 228
    assert plan["unique_invocations"] == len(plan["cases"]) == 220
    for case in plan["cases"]:
        assert sum(case["expert_counts"]) == case["received_tokens"] <= case["physical_rows"]
        if case["received_tokens"] == 1:
            assert case["distributions"] == ["balanced", "single_expert"]
        if case["phase"] == "decode":
            assert case["physical_rows"] == 4096 if case["local_experts"] == 16 else case["physical_rows"] == 8192
        else:
            assert case["received_tokens"] == case["physical_rows"]
        if case["distributions"] == ["single_expert"]:
            assert case["expert_counts"][1:] == [0] * (case["local_experts"] - 1)
    (tmp_path / "quant_model_description.json").write_text("{}")
    with pytest.raises(ValueError, match="dynamic W8A8"):
        get_ascend_moe_chain_cases(tmp_path)
