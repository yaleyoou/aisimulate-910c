import json

import pytest

from aisimulate_core.sdk.config_builders import build_model_config
from aisimulate_core.sdk.engine import _engine_config_dict
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError, PerfDataNotAvailableError
from aisimulate_core.sdk.fpm_config import OpLevelCompileConfig
from aisimulate_core.sdk.models import get_model
from tests.unit.sdk.ascend_fixtures import write_glm52_modelslim

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    return write_glm52_modelslim(tmp_path_factory.mktemp("ascend-forward-model"))


def configuration():
    return build_model_config(
        tp_size=8,
        pp_size=1,
        attention_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=16,
        gemm_quant_mode="w8a8",
        moe_quant_mode="w8a8_dynamic",
        kvcache_quant_mode="bfloat16",
        system="ascend910c",
    )


def flattened(operations):
    output = {}

    def visit(spec):
        kind, value = next(iter(spec.items()))
        if kind == "Overlap":
            for child in value["group_a"] + value["group_b"]:
                visit(child)
        elif kind == "TokenScale":
            visit(value["op"])
        else:
            output[value["name"]] = (kind, value)

    for operation in operations:
        visit(json.loads(operation._spec_json()))
    return output


def test_resident_inventory_matches_loaded_united_parameters(model_path):
    model = get_model(model_path, configuration(), "sglang")
    assert model.get_resident_weights_bytes() == 52_471_997_948
    context, generation = flattened(model.context_ops), flattened(model.generation_ops)
    assert context["context_indexer"][1]["scale_factor"] == 21
    assert context["context_attention"][1]["scale_factor"] == 78
    assert context["context_dense_gate_up"][1]["scale_factor"] == 3
    assert context["context_routed_experts"][1]["scale_factor"] == 75
    assert context["context_rope_cache"][1]["scale_factor"] == 2
    assert context["context_rope_cache"][1]["resident_copies"] == 1
    assert context["context_dense_global_sums"][1]["scale_factor"] == 6
    assert generation["generation_dense_input_gather"][1]["scale_factor"] == 3
    assert generation["generation_dense_output_reduce_scatter"][1]["scale_factor"] == 3
    assert generation["generation_sparse_reduce_scatter"][1]["scale_factor"] == 75
    assert generation["generation_dense_attention_reduce_scatter"][1]["scale_factor"] == 3
    assert context["context_logits_gather"][1]["token_axis"] == "requests"
    assert context["context_logits_finalize"][1]["token_axis"] == "requests"


def test_decode_padding_preserves_global_dense_and_rank_local_expert_shapes(model_path):
    model = get_model(model_path, configuration(), "sglang")
    decoded = [json.loads(operation._spec_json()) for operation in model.generation_ops]
    dense = next(
        spec["TokenScale"]
        for spec in decoded
        if "TokenScale" in spec
        and spec["TokenScale"]["op"].get("AscendLinear", {}).get("name") == "generation_dense_gate_up"
    )
    assert dense["numerator"] == 2 and dense["denominator"] == 1
    assert dense["op"]["AscendLinear"]["token_multiple"] == 16
    ops = flattened(model.generation_ops)
    assert ops["generation_shared_gate_up"][1]["token_split"] == 8
    assert ops["generation_moe_dispatch"][1]["input_token_multiple"] == 8
    assert ops["generation_indexer_weights"][1]["key"]["quant_mode"] == "float32_bfloat16"
    assert ops["generation_final_norm"][1]["token_split"] == 1
    assert ops["generation_sparse_input_norms"][1]["token_split"] == 8


def test_explicit_serial_shared_experts_survive_saved_engine_identity(model_path):
    config = configuration()
    config.op_level_config = OpLevelCompileConfig(
        {"ascend": {"shared_expert_overlap": False}}, system="ascend910c", backend="sglang"
    )
    model = get_model(model_path, config, "sglang")
    assert all("Overlap" not in json.loads(operation._spec_json()) for operation in model.generation_ops)
    assert model.get_resident_weights_bytes() == 52_471_997_948
    engine = _engine_config_dict(
        model=model,
        model_path=model_path,
        system="ascend910c",
        backend="sglang",
        backend_version="0.5.22",
        kv_block_size=128,
        systems_path=None,
        nextn=0,
    )
    saved = json.loads(engine["extra"]["estimator_config"])
    assert saved["op_level"]["ascend"] == {
        "shared_expert_overlap": False,
        "routing": "balanced",
        "kv_transfer_page_layout": "contiguous",
    }


def test_model_name_and_w8a8_label_cannot_replace_checkpoint_metadata():
    with pytest.raises(InvalidEngineConfigurationError, match="quant_model_description"):
        get_model("zai-org/GLM-5.2", configuration(), "sglang")


def transfer_model(model_path, role, **overrides):
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    return RustForwardPassPerfModel.best_available(
        {
            "model": model_path,
            "system": "ascend910c",
            "backend": "sglang",
            "backend_version": "0.5.22",
            "worker_type": role,
            "tp": 8,
            "attention_dp": 2,
            "moe_tp_size": 1,
            "moe_ep_size": 16,
            "kv_block_size": 128,
            "gemm_quant_mode": "w8a8",
            "moe_quant_mode": "w8a8_dynamic",
            "kvcache_quant_mode": "bfloat16",
            "fmha_quant_mode": "bfloat16",
            "estimation_mode": "op_level",
            **overrides,
        }
    )


def test_canonical_pd_model_selects_measured_mapping_and_rejects_unknown_geometry(model_path):
    source = transfer_model(model_path, "prefill")
    target = transfer_model(model_path, "decode")
    metadata = source.kv_transfer_metadata(target)
    assert metadata["key"] == {"source_tp": 8, "destination_tp": 8, "page_layout": "contiguous"}
    assert metadata["bytes_per_token_per_replica"] == 109824
    assert source.estimate_kv_transfer_time_ms(target, 0) == 0
    latency = source.estimate_kv_transfer_time_ms(target, 8192)
    assert 7.7 < latency < 7.9
    assert source.estimate_kv_transfer_time_ms(target, 8191) == latency
    with pytest.raises(PerfDataNotAvailableError, match="outside measured"):
        source.estimate_kv_transfer_time_ms(target, 32768)
    with pytest.raises(ValueError, match="matching Ascend"):
        target.estimate_kv_transfer_time_ms(source, 8192)
    multi_node = transfer_model(model_path, "decode", attention_dp=4, moe_ep_size=32)
    with pytest.raises(ValueError, match="one 16-device node per role"):
        source.estimate_kv_transfer_time_ms(multi_node, 8192)
    fragmented = transfer_model(
        model_path,
        "decode",
        estimator_config={"op_level": {"ascend": {"kv_transfer_page_layout": "fragmented"}}},
    )
    with pytest.raises(ValueError, match="matching role page-layout"):
        source.estimate_kv_transfer_time_ms(fragmented, 8192)
