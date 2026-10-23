"""GLM ModelSlim composition from the pinned Ascend operator and boundary captures.

The United observations use attention-local TP, global dense TP and DeepEP EP.
Timing and weight arithmetic remain in the native operations. The supported
MoE estimate uses the explicitly recorded synthetic balanced routing profile.
"""

import json

import aisimulate_core._native as core
from aisimulate_core.sdk import common
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError
from aisimulate_core.sdk.fpm_config import OpLevelCompileConfig


def configure_ascend_glm52(model, raw_config):
    config = model.config
    geometry = {
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "rms_norm_eps": 1e-5,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "rope_interleave": True,
        "vocab_size": 154880,
        "max_position_embeddings": 1048576,
    }
    if not isinstance(raw_config, dict) or any(raw_config.get(name) != value for name, value in geometry.items()):
        raise InvalidEngineConfigurationError("Ascend GLM composition requires the audited GLM-5.2 geometry")
    if raw_config.get("index_topk_pattern") is not None or raw_config.get("hidden_act") != "silu":
        raise InvalidEngineConfigurationError("Unqualified Ascend indexer/activation policy")
    metadata = raw_config.get("modelslim_metadata")
    if not isinstance(metadata, dict) or metadata.get("schema") != "aisimulate-modelslim-metadata/v1":
        raise InvalidEngineConfigurationError(
            "Ascend GLM prediction requires the local checkpoint and its quant_model_description.json"
        )
    attention = dict(q_a_proj="W8A8", q_b_proj="W8A8", kv_a_proj_with_mqa="W8A8", kv_b_proj="FLOAT", o_proj="W8A8")
    projections = dict.fromkeys(("gate_proj", "up_proj", "down_proj"), "W8A8_DYNAMIC")
    indexer = dict(wq_b="W8A8", wk="FLOAT", weights_proj="FLOAT")
    layers = metadata.get("layers", [])
    if len(layers) != 78 or metadata.get("embedding") != "FLOAT" or metadata.get("lm_head") != "FLOAT":
        raise InvalidEngineConfigurationError("Incomplete Ascend precision metadata")
    for layer_id, layer in enumerate(layers):
        expected_indexer = indexer if layer_id < 3 or (layer_id - 2) % 4 == 0 else None
        expected_mlp = (
            {"kind": "dense", "projections": projections}
            if layer_id < 3
            else {"kind": "moe", "experts": 256, "routed": projections, "router": "FLOAT", "shared": projections}
        )
        if layer != {"layer": layer_id, "attention": attention, "indexer": expected_indexer, "mlp": expected_mlp}:
            raise InvalidEngineConfigurationError(f"Unqualified ModelSlim precision partition at layer {layer_id}")
    if (
        model._backend_name != "sglang"
        or model.architecture != "GlmMoeDsaForCausalLM"
        or model._num_layers != 78
        or config.pp_size != 1
        or config.cp_size != 1
        or (config.dcp_size or 1) != 1
        or config.nextn != 0
        or config.enable_eplb
        or config.wideep_num_slots is not None
        or config.moe_backend is not None
        or config.moe_kernel_source is not None
        or config.attention_backend is not None
        or config.gemm_quant_mode != common.GEMMQuantMode.w8a8
        or config.moe_quant_mode != common.MoEQuantMode.w8a8_dynamic
        or config.kvcache_quant_mode != common.KVCacheQuantMode.bfloat16
        or config.fmha_quant_mode != common.FMHAQuantMode.bfloat16
        or config.comm_quant_mode not in (None, common.CommQuantMode.half)
    ):
        raise InvalidEngineConfigurationError(
            "Ascend GLM requires the pinned SGLang ModelSlim path, BF16 attention/KV, "
            "PP1/CP1/DCP1 and no speculation/EPLB"
        )
    local_tp, attention_dp, experts_parallel = config.tp_size, config.attention_dp_size, config.moe_ep_size
    if (
        local_tp not in (4, 8)
        or experts_parallel not in (16, 32)
        or config.moe_tp_size != 1
        or local_tp * attention_dp != experts_parallel
    ):
        raise InvalidEngineConfigurationError(
            "Ascend GLM requires local attention TP4/8, EP16/32, MoE-TP1 and TP*ADP=EP"
        )
    if config.op_level_config is None:
        config.op_level_config = OpLevelCompileConfig(None, system=config.system, backend=model._backend_name)
    settings = config.op_level_config.options["ascend"]
    model.raw_config = raw_config
    model._num_moe_layers = 75
    model._gpus_per_node = 16
    model._mtp_scale_factor = 1.0
    model.execution_assumptions = {
        "routing": settings["routing"],
        "shared_expert_overlap": settings["shared_expert_overlap"],
        "batch_scope": "per_attention_dp_replica",
        "attention_dp_load": "uniform",
        "decode_graph_buckets": [8, 16, 24, 32],
        "kv_pool_tokens": 53248,
        "kv_page_size": 128,
        "backend_revision": "baa9befa381511cd4d56f83980951e5a45fffc6d",
    }
    heads, dense_intermediate, vocabulary = 64 // local_tp, 12288 // experts_parallel, 154880 // local_tp

    for context in (True, False):
        phase, mode = ("context", "eager") if context else ("generation", "graph")
        padding = 1 if context else 8
        entries = []

        def linear(
            name, output, input_width, quant, count=1, bias="never", split=1, global_tokens=False, requests=False
        ):
            operation = {
                "AscendLinear": {
                    "name": phase + "_" + name,
                    "scale_factor": float(count),
                    "key": {
                        "quant_mode": quant,
                        "n": output,
                        "k": input_width,
                        "weight_format": 2 if quant in ("bfloat16", "float32_bfloat16") else 29,
                        "execution_mode": mode,
                    },
                    "bias": bias,
                    "token_axis": "requests" if requests else "tokens",
                    "token_split": split,
                    "token_multiple": max(1, padding // split) * (attention_dp if global_tokens else 1),
                }
            }
            return (
                {"TokenScale": {"op": operation, "numerator": attention_dp, "denominator": 1}}
                if global_tokens
                else operation
            )

        def vector(name, role, dimensions, count=1, split=1, global_tokens=False, resident_copies=None, requests=False):
            operation = {
                "AscendVector": {
                    "name": phase + "_" + name,
                    "scale_factor": float(count),
                    "key": {"role": role, "dimensions": dimensions, "execution_mode": mode},
                    "token_axis": "requests" if requests else "tokens",
                    "token_split": split,
                    "token_multiple": max(1, padding // split) * (attention_dp if global_tokens else 1),
                    "resident_copies": resident_copies,
                }
            }
            return (
                {"TokenScale": {"op": operation, "numerator": attention_dp, "denominator": 1}}
                if global_tokens
                else operation
            )

        def collective(name, operation, ranks, count=1, shard=False, global_tokens=False, width=6144, requests=False):
            elements = width * (attention_dp if global_tokens else 1)
            split = local_tp if shard else 1
            if not context:
                elements *= 8 // split
                split = 8
            return {
                "Hccl": {
                    "name": phase + "_" + name,
                    "scale_factor": float(count),
                    "key": {
                        "dtype": "bfloat16",
                        "operation": operation,
                        "num_ranks": ranks,
                        "ranks_per_node": min(ranks, 16),
                        "rank_layout": "contiguous",
                        "execution_mode": mode,
                    },
                    "elements_per_token": elements,
                    "token_split": split,
                    "token_axis": "requests" if requests else "tokens",
                }
            }

        def attention_op(name, operation, count, local_heads):
            return {
                "AscendAttention": {
                    "name": phase + "_" + name,
                    "scale_factor": float(count),
                    "key": {
                        "operation": operation,
                        "phase": "prefill" if context else "decode",
                        "local_heads": local_heads,
                        "execution_mode": mode,
                    },
                    "graph_buckets": [] if context else [8, 16, 24, 32],
                }
            }

        def transport(operation):
            return {
                "AscendDeepEp": {
                    "name": phase + "_moe_" + operation,
                    "scale_factor": 75.0,
                    "key": {
                        "phase": "normal" if context else "low_latency",
                        "operation": operation,
                        "num_ranks": experts_parallel,
                        "ranks_per_node": 16,
                        "distribution": settings["routing"],
                        "execution_mode": mode,
                    },
                    "token_split": local_tp,
                    "input_token_multiple": padding,
                }
            }

        entries.extend(
            [
                vector("embedding", "embedding", [vocabulary, 6144]),
                collective("embedding_all_reduce", "all_reduce", local_tp),
                vector("first_norm", "rms_bias", [6144]),
                vector("dense_input_norms", "add_rms_bias", [6144], 3),
                vector("sparse_input_norms", "add_rms_bias", [6144], 74, split=local_tp),
                vector("rope_cache", "rope_cache_gather", [1048576, 64], 2, resident_copies=1),
                linear("qkv_a", 2624, 6144, "w8a8", 78, "always"),
                vector("qk_norm", "split_qk_norm", [2048, 512, 64], 78),
                linear("q_b", heads * 256, 2048, "w8a8", 78, "always"),
                vector("rope_attention", "rope_attention", [heads, 64, 256, 64], 78),
                vector("cache_latent", "cache_scatter", [53376, 512], 78),
                vector("cache_rope", "cache_scatter", [53376, 64], 78),
                linear("indexer_query", 4096, 2048, "w8a8", 21, "always"),
                linear("indexer_key", 128, 6144, "bfloat16", 21),
                vector("indexer_norm", "indexer_norm", [128], 21),
                vector("rope_indexer", "rope_indexer", [32, 64, 128, 128], 21),
                vector("indexer_concat_query", "concat_indexer_q", [32, 64, 128], 21),
                vector("indexer_concat_key", "concat_indexer_k", [1, 64, 128], 21),
                vector("indexer_input_cast", "cast_hidden_fp32", [6144], 21),
                linear("indexer_weights", 32, 6144, "float32_bfloat16", 21),
                vector("indexer_weights_cast", "cast_weights_bf16", [32], 21),
                vector("cache_indexer", "cache_scatter", [53376, 128], 21),
                attention_op("indexer", "lightning_indexer", 21, 32),
                attention_op("attention", "sparse_attention", 78, heads),
            ]
        )
        for role in ("query_nope", "output_prefill" if context else "output_decode"):
            entries.append(
                {
                    "AscendBmm": {
                        "name": phase + "_mla_" + role,
                        "scale_factor": 78.0,
                        "key": {"role": role, "local_heads": heads, "execution_mode": mode},
                        "token_multiple": padding,
                        "resident_copies": 156,
                    }
                }
            )
        entries.extend(
            [
                linear("attention_output", 6144, heads * 256, "w8a8", 78, "tp_group_max"),
                vector("dense_residual_add", "shared_add", [6144], 3),
                vector("dense_post_norm", "rms_bias", [6144], 3, global_tokens=True),
                linear("dense_gate_up", 2 * dense_intermediate, 6144, "w8a8_dynamic", 3, global_tokens=True),
                vector("dense_activation", "swiglu", [dense_intermediate], 3, global_tokens=True),
                linear("dense_down", 6144, dense_intermediate, "w8a8_dynamic", 3, global_tokens=True),
                collective("sparse_reduce_scatter", "reduce_scatter", local_tp, 75),
                vector("sparse_post_norm", "add_rms_bias", [6144], 75, split=local_tp),
                linear("router", 256, 6144, "bfloat16", 75, split=local_tp),
            ]
        )
        if context:
            entries.append(collective("dense_global_sums", "all_reduce", experts_parallel, 6, global_tokens=True))
        else:
            entries.extend(
                [
                    collective("dense_attention_reduce_scatter", "reduce_scatter", local_tp, 3),
                    collective("dense_input_gather", "all_gather", experts_parallel, 3, shard=True),
                    collective(
                        "dense_output_reduce_scatter", "reduce_scatter", experts_parallel, 3, global_tokens=True
                    ),
                    collective("dense_output_gather", "all_gather", local_tp, 3, shard=True),
                ]
            )
        routed = [
            vector("router_topk", "router_topk", [256, 8], 75, split=local_tp),
            transport("dispatch"),
            {
                "AscendMoeChain": {
                    "name": phase + "_routed_experts",
                    "scale_factor": 75.0,
                    "key": {
                        "phase": "prefill" if context else "decode",
                        "local_experts": 256 // experts_parallel,
                        "distribution": settings["routing"],
                        "execution_mode": mode,
                    },
                    "token_numerator": 8 * attention_dp,
                    "token_denominator": experts_parallel,
                    "input_token_multiple": padding,
                }
            },
            transport("combine"),
            vector("routed_scale", "routed_scale", [6144], 75, split=local_tp),
        ]
        shared = [
            linear("shared_gate_up", 4096, 6144, "w8a8_dynamic", 75, split=local_tp),
            vector("shared_activation", "swiglu", [2048], 75, split=local_tp),
            linear("shared_down", 6144, 2048, "w8a8_dynamic", 75, split=local_tp),
        ]
        if settings["shared_expert_overlap"]:
            entries.append({"Overlap": {"name": phase + "_moe_overlap", "group_a": routed, "group_b": shared}})
        else:
            entries.extend([*routed, *shared])
        entries.extend(
            [
                vector("shared_add", "shared_add", [6144], 75, split=local_tp),
                vector("final_residual_add", "shared_add", [6144], split=local_tp),
                collective("sparse_output_gather", "all_gather", local_tp, 75, shard=True),
                vector("final_norm", "rms_bias", [6144]),
                linear("logits_gemm", vocabulary, 6144, "bfloat16", requests=True),
                collective("logits_gather", "all_gather", local_tp, width=vocabulary, requests=True),
                vector("logits_finalize", "logits_finalize", [local_tp, 154880], requests=True),
            ]
        )
        operations = [core.op_from_spec_json(json.dumps(entry)) for entry in entries]
        if context:
            model.context_ops = operations
        else:
            model.generation_ops = operations
