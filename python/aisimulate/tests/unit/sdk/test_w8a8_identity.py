import pytest

from aisimulate_core.quantization import GEMMQuantMode, MoEQuantMode
from aisimulate_core.sdk.rust_engine_step import _moe_quant_to_dtype, _quant_to_dtype

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", [GEMMQuantMode.w8a8, GEMMQuantMode.w8a8_dynamic])
def test_w8a8_keeps_distinct_int8_arithmetic_identity(mode):
    assert mode.value.memory == 1
    assert mode.value.compute_dtype == "int8"
    assert _quant_to_dtype(mode) == mode.name
    assert mode is not GEMMQuantMode.int8_wo
    assert mode is not GEMMQuantMode.sq


def test_moe_dynamic_int8_survives_engine_identity_conversion():
    mode = MoEQuantMode.w8a8_dynamic
    assert mode.value.memory == 1
    assert mode.value.compute_dtype == "int8"
    assert _moe_quant_to_dtype(mode) == "w8a8_dynamic"
