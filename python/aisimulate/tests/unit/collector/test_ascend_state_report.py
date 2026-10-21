import json

import pytest

from collector.sglang.ascend_state_report import summarize

pytestmark = pytest.mark.unit


def write_capture(directory):
    for rank, counts in enumerate([[1, 0], [0, 2]]):
        layers = [
            {
                "parameter": f"model.layers.{layer}.mlp.experts.w13_weight",
                "input_shape": [8, 4],
                "input_dtype": "torch.int8",
                "weight_shape": [2, 4, 6],
                "weight_dtype": "torch.int8",
                "output_dtype": "torch.bfloat16",
                "group_list_type": 1,
                "expert_counts": counts,
            }
            for layer in range(2)
        ]
        record = {
            "rank": rank,
            "ordinal": 0,
            "execution": "graph",
            "forward_mode": "DECODE",
            "seq_lens": [9],
            "input_tokens": 1,
            "layers": layers,
        }
        (directory / f"rank-{rank}.jsonl").write_text(json.dumps(record) + "\n")


def test_group_rows_are_counts_not_buffer_capacity(tmp_path):
    write_capture(tmp_path)
    report = summarize(tmp_path, ranks=2, layers=2, dense_layers=0, experts=4)
    assert report["forwards"][0]["group_rows_per_layer"]["0"] == {"total": 3, "maximum_rank": 2, "minimum_rank": 1}
    assert report["unique_gmm1_cases"] == 2
    assert all(case["input_shape"][0] == 8 for case in report["cases"])


def test_missing_rank_and_layer_are_not_partial_success(tmp_path):
    write_capture(tmp_path)
    (tmp_path / "rank-1.jsonl").unlink()
    with pytest.raises(ValueError, match="every expected rank"):
        summarize(tmp_path, ranks=2, layers=2, dense_layers=0, experts=4)
    write_capture(tmp_path)
    path = tmp_path / "rank-0.jsonl"
    value = json.loads(path.read_text())
    value["layers"].pop()
    path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="Incomplete model-layer"):
        summarize(tmp_path, ranks=2, layers=2, dense_layers=0, experts=4)


def test_invalid_counts_cannot_become_benchmark_cases(tmp_path):
    write_capture(tmp_path)
    path = tmp_path / "rank-0.jsonl"
    value = json.loads(path.read_text())
    value["layers"][0]["expert_counts"] = [-1, 1]
    path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="nonnegative integers"):
        summarize(tmp_path, ranks=2, layers=2, dense_layers=0, experts=4)
