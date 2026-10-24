import json
from pathlib import Path

import pytest
from collector.sglang.ascend_dcp_attention_cases import get_ascend_dcp_attention_cases

pytestmark = pytest.mark.unit


def test_npu_page_ownership_and_graph_layout_match_real_dcp_capture(tmp_path):
    config = Path(__file__).resolve().parents[3] / "src/aisimulate_core/model_configs/zai-org--GLM-5.2_config.json"
    (tmp_path / "config.json").write_bytes(config.read_bytes())
    plan = get_ascend_dcp_attention_cases(tmp_path)
    assert plan["boundary_contract"] == "sglang_npu_dcp_storage_attention_chain/v1"
    assert len(plan["cases"]) == 2736
    assert {case["kv_interleave_size"] for case in plan["cases"]} == {128}
    assert len({json.dumps(case, sort_keys=True) for case in plan["cases"]}) == 2736
    decoded = next(
        case
        for case in plan["cases"]
        if case["dcp_size"] == 2
        and case["local_heads"] == 8
        and case["phase"] == "decode"
        and case["active_batch"] == 1
        and case["sequence_length"] == 8192
    )
    assert decoded["padded_batch"] == 8 and decoded["block_table_width"] == 704
    prefill = next(
        case
        for case in plan["cases"]
        if case["dcp_size"] == 2
        and case["local_heads"] == 8
        and case["phase"] == "prefill"
        and case["active_batch"] == 1
        and case["tokens"] == 4096
        and case["prefix"] == 0
    )
    assert prefill["block_table_width"] == 16
    assert all(case["local_heads"] * case["dcp_size"] <= 64 for case in plan["cases"])
    altered = json.loads(config.read_text())
    altered["index_topk"] = 4096
    (tmp_path / "config.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="index_topk"):
        get_ascend_dcp_attention_cases(tmp_path)
