import json

import pytest

from collector.sglang.ascend_bmm_cases import get_ascend_bmm_cases

pytestmark = pytest.mark.unit


def test_all_roles_preserve_the_observed_mla_views(tmp_path):
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_attention_heads": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    plan = get_ascend_bmm_cases(tmp_path)
    assert len(plan["cases"]) == 222
    assert len({json.dumps(case, sort_keys=True) for case in plan["cases"]}) == 222
    for case in plan["cases"]:
        if case["role"] == "output_decode":
            assert case["tokens"] <= 32
        if case["role"] == "query_nope":
            assert case["input_stride"] == [case["local_heads"] * 256, 256, 1]
            assert case["weight_stride"] == [98304, 1, 192]
            assert case["output_width"] == 512
        else:
            assert case["input_stride"] == [case["local_heads"] * 512, 512, 1]
            assert case["weight_stride"] == [131072, 256, 1]
            assert case["output_width"] == 256
        assert case["kernel_source"] == (
            "torch.ops.npu.batch_matmul_transpose"
            if case["role"] == "output_decode"
            else "torch_npu.npu_transpose_batchmatmul"
        )
    config["qk_rope_head_dim"] = 32
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="qk_rope_head_dim"):
        get_ascend_bmm_cases(tmp_path)
