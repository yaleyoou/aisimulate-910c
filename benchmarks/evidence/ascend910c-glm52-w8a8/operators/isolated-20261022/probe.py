"""Experimental framework-dispatched Ascend W8A8 numerical/timing probes.

Synthetic inputs qualify these kernels only; routed group lists are declared
balanced controls, not substitutes for serving routing or canonical FPM rows.
"""

import argparse
import gc
import importlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--graph-calls", type=int, default=16)
    parser.add_argument("--family", choices=("quant", "attention"), default="quant")
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.graph_calls) < 1:
        parser.error("warmup, repeats and graph-calls must be positive")
    import torch
    import torch_npu

    importlib.import_module("sglang.srt.layers.quantization")

    from sglang.srt.hardware_backend.npu.moe.quant import HiddenStatesDynamicQuant
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend, npu_format_cast
    from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w8a8_int8 import ModelSlimW8A8Int8

    init_npu_backend()
    torch.npu.set_device(args.device)
    torch.manual_seed(910)
    result = {
        "schema": "aisimulate-ascend-quant-probe/v1",
        "canonical_fpm_data": False,
        "input_kind": "seeded synthetic activations and INT8 weights",
        "group_list": "balanced synthetic cumulative group sizes; not observed routing",
        "timing": {
            "clock": "NPU events around graph replay, divided by calls in captured graph",
            "warmup": args.warmup,
            "warmup_scope": "Both eager invocation and captured graph replay",
            "independent_replays": args.repeats,
            "calls_per_graph": args.graph_calls,
            "scope": "Device execution including graph dependencies; excludes eager Python launch overhead.",
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def compare(actual, expected):
        difference = (actual.float().cpu() - expected).abs()
        tolerance = 0.02 + 0.012 * expected.abs()
        if not torch.all(difference <= tolerance):
            raise ValueError(f"CPU reference mismatch: max error={difference.max().item()}")
        return {"max_abs_error": difference.max().item(), "samples": actual.numel(), "passed": True}

    def measure(function):
        for _ in range(args.warmup):
            function()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(args.graph_calls):
                function()
        for _ in range(args.warmup):
            graph.replay()
        torch.npu.synchronize()
        samples = []
        for _ in range(args.repeats):
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / args.graph_calls)
        return samples

    def dynamic_quant(tokens, hidden):
        activations = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16)
        kernel = HiddenStatesDynamicQuant(torch.int8)
        function = lambda: kernel(activations)
        quantized, scale = function()
        row_ids = sorted({0, tokens // 2, tokens - 1})
        original = activations[row_ids].float().cpu()
        expected_scale = original.abs().amax(dim=-1) / 127
        torch.testing.assert_close(scale[row_ids].float().cpu(), expected_scale, rtol=1e-5, atol=1e-7)
        expected = (original / expected_scale[:, None]).round().clamp(-128, 127)
        delta = (quantized[row_ids].float().cpu() - expected).abs().max().item()
        if delta > 1:
            raise ValueError(f"Dynamic quantization differs from CPU reference by {delta} bins")
        record(
            {"op": "dynamic_quant", "m": tokens, "k": hidden}, function,
            {"passed": True, "max_int8_bin_difference": delta, "sampled_elements": original.numel()},
        )

    def record(identity, function, correctness):
        samples = measure(function)
        cv = statistics.pstdev(samples) / statistics.mean(samples)
        row = {
            **identity,
            "correctness": correctness,
            "samples_us": samples,
            "median_us": statistics.median(samples),
            "cv": cv,
            "repeatable_at_10_percent": cv <= 0.10,
        }
        result["cases"].append(row)
        save()
        print(json.dumps({**identity, "median_us": row["median_us"], "cv": cv}), flush=True)

    def linear(tokens, input_dim, output_dim, dynamic):
        activations = torch.randn(tokens, input_dim, device="npu", dtype=torch.bfloat16)
        weights = torch.randint(-3, 4, (input_dim, output_dim), device="npu", dtype=torch.int8)
        layer = torch.nn.Module()
        mode = "W8A8_DYNAMIC" if dynamic else "W8A8"
        kernel = ModelSlimW8A8Int8({"probe.weight": mode}, "probe")
        with torch.device("npu"):
            kernel.create_weights(layer, input_dim, [output_dim], input_dim, output_dim, torch.bfloat16)
        layer.weight.data.copy_(weights.transpose(0, 1))
        layer.weight_scale.data.fill_(0.125)
        layer.weight_offset.data.zero_()
        if dynamic:
            quantized, scales = torch_npu.npu_dynamic_quant(activations)
        else:
            layer.input_scale.data.fill_(1 / 32)
            layer.input_offset.data.zero_()
            layer.deq_scale.data.fill_(0.125 / 32)
            layer.quant_bias.data.zero_()
            quantized = (activations.float().cpu() * 32).round().clamp(-128, 127).to(torch.int8)
            scales = torch.full((tokens,), 1 / 32)
        kernel.process_weights_after_loading(layer)
        row_ids = sorted({0, tokens // 2, tokens - 1})
        column_ids = list(range(0, output_dim, max(1, output_dim // 32)))
        expected = quantized[row_ids].float().cpu() @ weights[:, column_ids].float().cpu()
        expected *= scales[row_ids].float().cpu().reshape(-1, 1) * 0.125
        function = lambda: kernel.apply_weights(layer, activations)
        checked = compare(function()[row_ids][:, column_ids], expected)
        record(
            {"op": "dynamic_linear" if dynamic else "static_linear", "m": tokens, "k": input_dim, "n": output_dim},
            function,
            checked,
        )

    def grouped(tokens, input_dim, output_dim):
        experts = 16
        activations = torch.randn(tokens, input_dim, device="npu", dtype=torch.bfloat16)
        quantizer = HiddenStatesDynamicQuant(torch.int8)
        quantized, scales = quantizer(activations)
        weights = torch.randint(-3, 4, (experts, input_dim, output_dim), device="npu", dtype=torch.int8)
        layer = SimpleNamespace(
            w13_weight=npu_format_cast(weights),
            w13_weight_scale=torch.full((experts, output_dim), 0.125, device="npu", dtype=torch.bfloat16),
        )
        sizes = [tokens // experts + int(index < tokens % experts) for index in range(experts)]
        endpoints = torch.tensor(sizes, device="npu", dtype=torch.int64).cumsum(0)
        kernel = NPUW8A8Int8MoEMethod()
        function = lambda: kernel.apply(layer, quantized, endpoints, scales, torch.bfloat16, "w13", 0)
        output = function()
        columns = list(range(0, output_dim, max(1, output_dim // 32)))
        checks = []
        begin = 0
        for expert, size in enumerate(sizes):
            if size:
                selected = [begin, begin + size - 1]
                expected = quantized[selected].float().cpu() @ weights[expert, :, columns].float().cpu()
                expected *= scales[selected].float().cpu().reshape(-1, 1) * 0.125
                checks.append(compare(output[selected][:, columns], expected))
            begin += size
        record(
            {"op": "grouped_matmul", "m": tokens, "k": input_dim, "n": output_dim, "experts": experts, "group_sizes": sizes},
            function,
            {"passed": True, "max_abs_error": max(check["max_abs_error"] for check in checks), "samples": sum(check["samples"] for check in checks)},
        )

    def attention(phase, sequence_length):
        tokens = 8192 if phase == "prefill" else 8
        batch = 1 if phase == "prefill" else 8
        pages = (sequence_length + 127) // 128
        block_table = torch.arange(batch * pages, device="npu", dtype=torch.int32).reshape(batch, pages)
        query_lengths = torch.tensor([tokens], device="npu", dtype=torch.int32) if batch == 1 else torch.arange(
            1, batch + 1, device="npu", dtype=torch.int32
        )
        key_lengths = torch.full((batch,), sequence_length, device="npu", dtype=torch.int32)
        index_query = torch.randn(tokens, 32, 128, device="npu", dtype=torch.bfloat16)
        index_key = torch.randn(batch * pages, 128, 1, 128, device="npu", dtype=torch.bfloat16)
        index_weights = torch.rand(tokens, 32, device="npu", dtype=torch.bfloat16)

        def indexer():
            return torch_npu.npu_lightning_indexer(
                query=index_query,
                key=index_key,
                weights=index_weights,
                actual_seq_lengths_query=query_lengths,
                actual_seq_lengths_key=key_lengths,
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )[0]

        indices = indexer()
        selected_rows = sorted({0, tokens // 2, tokens - 1})
        overlaps = []
        for row in selected_rows:
            visible = min(sequence_length, row + 1) if phase == "prefill" else sequence_length
            key_batch = 0 if phase == "prefill" else row
            key_cpu = index_key.reshape(batch, pages * 128, 128)[key_batch, :visible].float().cpu()
            scores = (index_query[row].float().cpu() @ key_cpu.T).relu()
            scores = (scores * index_weights[row].float().cpu().reshape(-1, 1)).sum(0)
            expected = set(scores.topk(min(2048, visible)).indices.tolist())
            observed_list = [value for value in indices[row].flatten().cpu().tolist() if value >= 0]
            observed = set(observed_list)
            overlap = len(expected & observed) / len(expected)
            if len(observed) != len(expected) or len(observed) != len(observed_list) or overlap < 0.99:
                raise ValueError(f"Indexer reference mismatch: row={row}, overlap={overlap}")
            if any(value >= visible for value in observed):
                raise ValueError("Indexer emitted a future or out-of-range token")
            overlaps.append(overlap)
        record(
            {"op": "lightning_indexer", "phase": phase, "tokens": tokens, "sequence_length": sequence_length},
            indexer,
            {"passed": True, "sampled_queries": len(selected_rows), "min_topk_overlap": min(overlaps), "threshold": 0.99},
        )
        query = torch.randn(tokens, 8, 512, device="npu", dtype=torch.bfloat16)
        query_rope = torch.randn(tokens, 8, 64, device="npu", dtype=torch.bfloat16)
        key = torch.randn(batch * pages, 128, 1, 512, device="npu", dtype=torch.bfloat16)
        key_rope = torch.randn(batch * pages, 128, 1, 64, device="npu", dtype=torch.bfloat16)

        def sparse_attention():
            return torch_npu.npu_sparse_flash_attention(
                query=query, key=key, value=key, query_rope=query_rope, key_rope=key_rope,
                sparse_indices=indices, scale_value=0.0625,
                actual_seq_lengths_query=query_lengths, actual_seq_lengths_kv=key_lengths,
                block_table=block_table, sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND",
                sparse_mode=3, attention_mode=2, return_softmax_lse=False,
            )[0]

        actual = sparse_attention()
        checks = []
        for row in selected_rows:
            key_batch = 0 if phase == "prefill" else row
            selected = indices[row].flatten().cpu()
            selected = selected[selected >= 0].long()
            keys = key.reshape(batch, pages * 128, 512)[key_batch].float().cpu()[selected]
            rope = key_rope.reshape(batch, pages * 128, 64)[key_batch].float().cpu()[selected]
            logits = query[row].float().cpu() @ keys.T + query_rope[row].float().cpu() @ rope.T
            expected = (logits * 0.0625).softmax(-1) @ keys
            checks.append(compare(actual[row], expected))
        record(
            {"op": "sparse_attention", "phase": phase, "tokens": tokens, "sequence_length": sequence_length,
             "scale": 0.0625, "scale_scope": "Synthetic numerical control, not loaded model RoPE scaling."},
            sparse_attention,
            {"passed": True, "max_abs_error": max(check["max_abs_error"] for check in checks),
             "samples": sum(check["samples"] for check in checks)},
        )

    try:
        for input_dim, output_dim in ([(6144, 2624), (2048, 2048), (2048, 4096), (2048, 6144)] if args.family == "quant" else []):
            for tokens in [8, 8192]:
                linear(tokens, input_dim, output_dim, False)
                gc.collect()
                torch.npu.empty_cache()
        for input_dim, output_dim, token_counts in ([
            (6144, 1536, [16, 8192]), (768, 6144, [16, 8192]),
            (6144, 4096, [1, 1024]), (2048, 6144, [1, 1024]),
        ] if args.family == "quant" else []):
            for tokens in token_counts:
                linear(tokens, input_dim, output_dim, True)
                gc.collect()
                torch.npu.empty_cache()
        for tokens in ([16, 256, 4096] if args.family == "quant" else []):
            for input_dim, output_dim in [(6144, 4096), (2048, 6144)]:
                grouped(tokens, input_dim, output_dim)
                gc.collect()
                torch.npu.empty_cache()
        if args.family == "quant":
            for tokens, hidden in [(1, 6144), (8, 6144), (1024, 6144), (8192, 6144), (8192, 768), (1024, 2048)]:
                dynamic_quant(tokens, hidden)
        if args.family == "attention":
            for phase, sequence_length in [("prefill", 8192), ("decode", 8192), ("decode", 8704), ("decode", 9216)]:
                attention(phase, sequence_length)
                gc.collect()
                torch.npu.empty_cache()
        result["status"] = "completed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
