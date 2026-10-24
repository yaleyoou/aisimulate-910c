import pytest

import aisimulate.sweeper.model_hw as model_hardware
from aisimulate.sweeper.model_hw import ModelHardware, NoViableParallelConfig, parallel_configs_for
from aisimulate.sweeper.parallel_enum import ParallelShape


@pytest.fixture
def declared_shapes(monkeypatch):
    shapes = (ParallelShape(4, 4, 1, 16), ParallelShape(8, 2, 1, 16))
    hardware = ModelHardware(
        model_name="model",
        hardware_sku="ascend910c",
        backend="sglang",
        is_moe=True,
        mla=True,
        enable_wideep=True,
        weight_bytes=1,
        vram_per_gpu=64 * 1024**3,
        gpus_per_node=16,
        max_context=9216,
        parallel_shape_candidates=shapes,
    )
    monkeypatch.setattr(model_hardware, "resolve_model_hardware", lambda *args, **kwargs: hardware)
    monkeypatch.setattr(
        model_hardware, "feasible_shape_tokens", lambda candidates, **kwargs: dict.fromkeys(candidates, 53248)
    )
    return shapes


def configurations(mode, budget, **kwargs):
    return parallel_configs_for(
        "model", "ascend910c", gpu_budget=budget, deployment_mode=mode, backend="sglang", max_seq_len=9216, **kwargs
    )


def test_automatic_menu_covers_all_replica_ratios_within_budget(declared_shapes):
    aggregated = configurations("agg", 64)
    assert len(aggregated) == 8
    assert {candidate.replicas for candidate in aggregated} == {1, 2, 3, 4}
    disaggregated = configurations("disagg", 64)
    assert len(disaggregated) == 24
    assert {(candidate.prefill.replicas, candidate.decode.replicas) for candidate in disaggregated} == {
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (2, 2),
        (3, 1),
    }
    assert all(candidate.total_gpus <= 64 for candidate in disaggregated)


def test_automatic_menu_obeys_smaller_budgets_and_reports_no_fit(declared_shapes):
    assert len(configurations("agg", 16)) == 2
    assert len(configurations("disagg", 32)) == 4
    assert all(candidate.total_gpus == 64 for candidate in configurations("disagg", 64, min_gpu_budget=64))
    with pytest.raises(NoViableParallelConfig):
        configurations("disagg", 16)


def test_declared_menu_still_uses_per_role_native_memory_feasibility(declared_shapes, monkeypatch):
    calls = []

    def feasible(candidates, **kwargs):
        calls.append(kwargs)
        return {shape: 53248 for shape in candidates if shape.tp == 8}

    monkeypatch.setattr(model_hardware, "feasible_shape_tokens", feasible)
    selected = configurations("disagg", 64, role_runtime={"prefill": (4096, 1, 0.9), "decode": (8192, 32, 0.9)})
    assert len(selected) == 6
    assert all(candidate.prefill.shape.tp == candidate.decode.shape.tp == 8 for candidate in selected)
    assert [(call["max_num_tokens"], call["max_batch_size"]) for call in calls] == [(4096, 1), (8192, 32)]
