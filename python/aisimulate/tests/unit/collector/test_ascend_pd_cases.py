import json
from pathlib import Path

import pytest
from collector.sglang.ascend_pd_cases import get_ascend_pd_cases

pytestmark = pytest.mark.unit


def test_pd_plan_preserves_replica_layout_and_page_capacity(tmp_path):
    source = Path(__file__).resolve().parents[3] / "src/aisimulate_core/model_configs/zai-org--GLM-5.2_config.json"
    (tmp_path / "config.json").write_bytes(source.read_bytes())
    (tmp_path / "quant_model_description.json").write_text(json.dumps({"model_quant_type": "W8A8"}))
    plan = get_ascend_pd_cases(tmp_path)
    cases = plan["cases"]
    assert len(cases) == len({json.dumps(case, sort_keys=True) for case in cases})
    assert len(cases) == 90
    assert {(case["source_tp"], case["destination_tp"]) for case in cases} == {
        (4, 4), (4, 8), (4, 16), (8, 4), (8, 8), (8, 16), (16, 4), (16, 8), (16, 16)
    }
    assert {case["page_layout"] for case in cases} == {"contiguous", "fragmented"}
    assert all(case["component_widths"] == [512, 64, 128] and case["dtype"] == "bfloat16" for case in cases)
    assert all(case["tokens"] % 128 == 0 and 2 * case["tokens"] <= 53248 for case in cases)
    config = json.loads(source.read_text())
    config["index_head_dim"] = 64
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="index_head_dim"):
        get_ascend_pd_cases(tmp_path)
