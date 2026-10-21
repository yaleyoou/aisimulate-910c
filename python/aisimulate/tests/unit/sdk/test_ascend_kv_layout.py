from dataclasses import replace
from types import SimpleNamespace

import pytest

from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate_core.sdk.common import KVCacheQuantMode
from aisimulate_core.sdk.config_builders import build_model_config
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError
from aisimulate_core.sdk.memory import kv_cache
from aisimulate_core.sdk.models import get_model

pytestmark = pytest.mark.unit


def model(system="ascend910c"):
    config = build_model_config(
        tp_size=8,
        pp_size=1,
        attention_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=16,
        gemm_quant_mode="w8a8",
        moe_quant_mode="w8a8_dynamic",
        kvcache_quant_mode="bfloat16",
        system=system,
    )
    return get_model("zai-org/GLM-5.2", config, "sglang")


def test_measured_pool_and_sentinel_reconcile():
    instance = model()
    assert instance.config.system == "ascend910c"
    assert instance.get_kvcache_bytes_per_sequence(1) == 109824
    assert instance.get_kvcache_pool_overhead_bytes() == 14057472
    assert instance.get_kvcache_bytes_per_sequence(52992) + instance.get_kvcache_pool_overhead_bytes() == 5833850880
    assert instance.get_kvcache_max_tokens(5833850880) == 52992
    assert instance.get_kvcache_rank_batch_capacity(5833850880, 10) == 52992
    assert instance.get_kvcache_max_tokens(5833850879) == 52864


def test_pool_overhead_is_once_per_pool():
    """One and two 9216-token requests need 73 and 145 measured 14057472-byte pages."""
    instance = model()
    database = SimpleNamespace(system_spec={"misc": {"nccl_mem": {8: 0}, "other_mem": 0}})
    backend = get_backend("sglang")
    one = backend._get_memory_usage(instance, database, batch_size=1, beam_width=1, isl=8192, osl=1024)
    two = backend._get_memory_usage(instance, database, batch_size=2, beam_width=1, isl=8192, osl=1024)
    assert one["kvcache"] * (1 << 30) == 1026195456
    assert two["kvcache"] * (1 << 30) == 2038333440


def test_no_capacity_until_sentinel_and_one_usable_page_fit():
    instance = model()
    for budget in (-1, 0, 14057472, 28114943):
        assert instance.get_kvcache_max_tokens(budget) == 0
    assert instance.get_kvcache_max_tokens(28114944) == 128
    assert instance.get_kvcache_bytes_per_sequence(0) == 0


def test_gpu_layout_preserves_fp8_indexer_and_linear_capacity():
    instance = model("h200_sxm")
    assert instance.get_kvcache_bytes_per_sequence(1) == 100152
    assert instance.get_kvcache_pool_overhead_bytes() == 0
    assert instance.get_kvcache_max_tokens(100152 * 129) == 129


@pytest.mark.parametrize(
    "changes",
    [
        {"pp_size": 2},
        {"cp_size": 2},
        {"dcp_size": 2},
        {"nextn": 1},
        {"kvcache_quant_mode": KVCacheQuantMode.fp8},
    ],
)
def test_unqualified_layouts_fail(changes):
    instance = model()
    instance.config = replace(instance.config, **changes)
    with pytest.raises(ValueError, match="Ascend910C DSA KV layout"):
        instance.get_kvcache_bytes_per_sequence(1)


def test_native_capacity_construction_preserves_system(monkeypatch):
    """Isolate cache geometry from resident overhead that still needs measurement."""
    system_spec = kv_cache.perf_database.load_system_spec("ascend910c")
    monkeypatch.setattr(
        kv_cache.perf_database,
        "get_database",
        lambda *_args, **_kwargs: SimpleNamespace(version="0.5.22", system_spec=system_spec),
    )
    monkeypatch.setattr(
        kv_cache,
        "get_backend",
        lambda _backend: SimpleNamespace(
            _get_memory_usage=lambda *_args, **_kwargs: dict(weights=1, activations=1, nccl=1, others=1)
        ),
    )
    estimator = kv_cache.KVCacheEstimator.from_request(
        "zai-org/GLM-5.2",
        "ascend910c",
        "sglang",
        "0.5.22",
        tp_size=8,
        attention_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=16,
        max_num_tokens=8192,
        max_batch_size=32,
        gemm_quant_mode="w8a8",
        moe_quant_mode="w8a8_dynamic",
        kvcache_quant_mode="bfloat16",
    )
    assert estimator.breakdown["kv_size_per_token_bytes"] == 109824
    assert estimator.breakdown["tokens_from_kv_bytes"](5833850880) == 52992


def test_naive_fallback_cannot_hide_invalid_layout(monkeypatch):
    def reject(*_args, **_kwargs):
        raise InvalidEngineConfigurationError("Ascend910C DSA KV layout is unqualified")

    monkeypatch.setattr(kv_cache.KVCacheEstimator, "from_request", reject)
    with pytest.raises(InvalidEngineConfigurationError, match="unqualified"):
        kv_cache.estimate_kv_cache(
            "zai-org/GLM-5.2",
            "ascend910c",
            "sglang",
            max_num_tokens=8192,
            max_batch_size=32,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            allow_naive_fallback=True,
        )
