import pytest
from collector.sglang.ascend_moe_compute import selected_rows, validate_case

pytestmark = pytest.mark.unit


def case():
    return {
        "input_shape": [4096, 6144],
        "weight_shape": [4, 6144, 4096],
        "input_dtype": "torch.int8",
        "weight_dtype": "torch.int8",
        "output_dtype": "torch.bfloat16",
        "group_list_type": 1,
        "execution": "graph",
        "expert_counts": [0, 14, 0, 14],
    }


def test_samples_only_populated_rows_in_padded_buffer():
    captured = case()
    assert validate_case(captured) == (4096, 6144, 2048, 4)
    assert selected_rows(captured["expert_counts"]) == [(1, 0), (1, 13), (3, 14), (3, 27)]


def test_empty_experts_and_single_row():
    captured = case()
    captured["expert_counts"] = [0, 0, 0, 0]
    validate_case(captured)
    assert selected_rows(captured["expert_counts"]) == []
    assert selected_rows([0, 1, 0]) == [(1, 0)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expert_counts", [0, 4097, 0, 0]),
        ("expert_counts", [0, -1, 0, 0]),
        ("expert_counts", [0, True, 0, 0]),
        ("expert_counts", [0, 1]),
        ("group_list_type", 0),
        ("input_shape", [4096, 2048]),
        ("weight_shape", [4, 6144, 4095]),
        ("input_dtype", "torch.bfloat16"),
        ("execution", "unknown"),
    ],
)
def test_rejects_changed_or_corrupt_invocation(field, value):
    captured = case()
    captured[field] = value
    with pytest.raises(ValueError, match="geometry"):
        validate_case(captured)
