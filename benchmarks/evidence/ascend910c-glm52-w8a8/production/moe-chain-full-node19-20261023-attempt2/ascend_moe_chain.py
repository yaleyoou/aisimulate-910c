"""Measure actual eager/graph expert compute with quantization reference checks."""

import argparse
import gc
import hashlib
import importlib
import json
import os
import socket
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--serving-setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--case-filter")
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.shard_index < args.num_shards:
        parser.error("Use a fresh output and a valid shard")
    plan = json.loads(args.plan.read_text())
    protocol = plan["protocol"]
    if (
        plan["schema"] != "aisimulate-ascend-moe-chain-plan/v1"
        or min(protocol.values()) < 1
        or protocol["repeats"] < 3
        or protocol["iterations"] % protocol["weight_copies"]
    ):
        raise ValueError("Invalid expert chain plan/protocol")
    setup = json.loads(args.serving_setup.read_text())
    for name, value in setup["environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    os.sched_setaffinity(0, setup["worker_cpu_affinity"][str(args.device)])
    import torch
    import torch_npu

    importlib.import_module("sglang.srt.layers.quantization")
    from sglang.srt.hardware_backend.npu.moe.activation import NPUSwigluDeepEPKernel
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend, npu_format_cast

    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    if revision != "baa9befa381511cd4d56f83980951e5a45fffc6d":
        raise ValueError("Reaudit expert dispatch for another runtime")
    init_npu_backend()
    torch.npu.set_device(args.device)
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    kernel, activation = NPUW8A8Int8MoEMethod(), NPUSwigluDeepEPKernel(need_quant=True)
    selected = [
        (index, case)
        for index, case in enumerate(plan["cases"])
        if not args.case_filter or args.case_filter in json.dumps(case, sort_keys=True)
    ]
    selected = [entry for position, entry in enumerate(selected) if position % args.num_shards == args.shard_index]
    if not selected:
        raise ValueError("Empty selected plan")
    record = {
        "schema": "aisimulate-ascend-moe-chain-observations/v1",
        "status": "running",
        "scope": plan["scope"],
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": protocol,
        "selected_plan_indices": [index for index, _ in selected],
        "cases": [],
        "runtime": {
            "framework_commit": revision,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(args.device),
            "host": socket.gethostname(),
            "device_index": args.device,
            "driver_version_info": Path("/usr/local/Ascend/driver/version.info").read_text(),
            "serving_setup": setup,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    def collect(index, case):
        torch.manual_seed(910 + index)
        experts, hidden, intermediate = case["local_experts"], case["hidden_size"], case["intermediate_size"]
        counts = case["expert_counts"]
        if (
            case["group_list_type"] != 1
            or len(counts) != experts
            or min(counts) < 0
            or sum(counts) != case["received_tokens"]
            or not 0 < sum(counts) <= case["physical_rows"]
        ):
            raise ValueError("Invalid count vector/buffer geometry")
        group = torch.tensor(counts, device="npu", dtype=torch.int64)
        values = torch.randint(-3, 4, (case["physical_rows"], hidden), device="npu", dtype=torch.int8)
        scales = torch.full((case["physical_rows"],), 1 / 32, device="npu", dtype=torch.float32)
        raw_up = torch.randint(-3, 4, (experts, hidden, intermediate * 2), device="npu", dtype=torch.int8)
        raw_down = torch.randint(-3, 4, (experts, intermediate, hidden), device="npu", dtype=torch.int8)
        weights = [
            SimpleNamespace(
                w13_weight=npu_format_cast(raw_up.clone()),
                w2_weight=npu_format_cast(raw_down.clone()),
                w13_weight_scale=torch.full((experts, intermediate * 2), 0.125, device="npu", dtype=torch.bfloat16),
                w2_weight_scale=torch.full((experts, hidden), 0.125, device="npu", dtype=torch.bfloat16),
            )
            for _ in range(protocol["weight_copies"])
        ]
        if any(
            torch_npu.get_npu_format(weight.w13_weight) != 29 or torch_npu.get_npu_format(weight.w2_weight) != 29
            for weight in weights
        ):
            raise ValueError("Expert weights must use the actual NZ format")
        references = []
        offset = 0
        columns = list(range(0, hidden, hidden // 32))
        for expert, count in enumerate(counts):
            if count:
                rows = sorted({offset, offset + count - 1})
                reference_input = values[rows].float().cpu()
                reference_up = reference_input @ raw_up[expert].float().cpu() / 256
                reference_down = raw_down[expert, :, columns].float().cpu()
                references.append((rows, reference_up, reference_down))
            offset += count
        del raw_up, raw_down

        def invoke(weight):
            gate = kernel.apply(weight, values, group, scales, torch.bfloat16, "w13", 1)
            quantized, quant_scales = activation._apply_activation(gate, group, 1)
            output = kernel.apply(weight, quantized, group, quant_scales, torch.bfloat16, "w2", 1)
            return gate, quantized, quant_scales, output

        def batch():
            return [invoke(weights[iteration % len(weights)]) for iteration in range(protocol["iterations"])]

        for _ in range(protocol["warmup"]):
            batch()
        torch.npu.synchronize()
        graph, outputs = None, None
        if case["execution_mode"] == "graph":
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = batch()
            for _ in range(protocol["warmup"]):
                graph.replay()
            torch.npu.synchronize()
        elif case["execution_mode"] != "eager":
            raise ValueError("Unsupported expert execution mode")

        def execute():
            if graph is None:
                return batch()
            graph.replay()
            return outputs

        sign = 1

        def verify():
            nonlocal sign
            sign = -sign
            values.neg_()
            observed = execute()
            torch.npu.synchronize()
            checked, maximum, max_bin_error, max_scale_error = 0, 0.0, 0.0, 0.0
            for rows, expected_up, down_weight in references:
                expected_up = (expected_up * sign).to(torch.bfloat16).float()
                for gate, quantized, quant_scales, output in observed:
                    gate_cpu = gate[rows].float().cpu()
                    error = (gate_cpu - expected_up).abs()
                    if not torch.all(error <= 0.02 + 0.012 * expected_up.abs()):
                        raise ValueError("MoE chain execution numerical check failed: GMM1")
                    gate_part, up_part = gate_cpu.chunk(2, dim=-1)
                    activated = torch.nn.functional.silu(gate_part) * up_part
                    expected_scales = activated.abs().amax(-1) / 127
                    actual_scales = quant_scales[rows].float().cpu()
                    if not torch.allclose(actual_scales, expected_scales, rtol=2e-4, atol=1e-7):
                        raise ValueError("MoE chain execution numerical check failed: quant scales")
                    expected_quant = (activated / expected_scales[:, None]).round().clamp(-128, 127)
                    actual_quant = quantized[rows].float().cpu()
                    bin_error = (actual_quant - expected_quant).abs().max().item()
                    if bin_error > 1:
                        raise ValueError("MoE chain execution numerical check failed: quant bins")
                    expected_down = (actual_quant @ down_weight) * actual_scales[:, None] * 0.125
                    down_error = (output[rows][:, columns].float().cpu() - expected_down).abs()
                    if not torch.all(down_error <= 0.02 + 0.012 * expected_down.abs()):
                        raise ValueError("MoE chain execution numerical check failed: GMM2")
                    checked += gate_cpu.numel() + actual_quant.numel() + down_error.numel()
                    maximum = max(maximum, error.max().item(), down_error.max().item())
                    max_bin_error = max(max_bin_error, bin_error)
                    max_scale_error = max(max_scale_error, (actual_scales - expected_scales).abs().max().item())
            return {
                "sampled_elements": checked,
                "max_abs_error": maximum,
                "max_quant_bin_error": max_bin_error,
                "max_quant_scale_error": max_scale_error,
            }

        checks = [verify()]
        samples, wall = [], []
        for _ in range(protocol["repeats"]):
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            result = execute()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / protocol["iterations"])
            wall.append((time.perf_counter() - started) * 1000 / protocol["iterations"])
            del result
        checks.append(verify())
        row = {
            **case,
            "plan_index": index,
            "checks": checks,
            "weight_format": 29,
            "weight_copies": len(weights),
            "input_dtype": "int8",
            "output_dtype": "bfloat16",
            "scale_dtypes": ["float32", "bfloat16"],
            "kernel_sources": {
                "gmm": type(kernel.matmul).__module__ + "." + type(kernel.matmul).__name__,
                "activation": activation._kernel.__module__ + "." + activation._kernel.__name__,
            },
            "execution_verification": "captured_graph_replay" if graph is not None else "eager_invocation",
            "samples_ms": samples,
            "wall_samples_ms": wall,
            "latency_ms": statistics.median(samples),
            "cv": statistics.pstdev(samples) / statistics.mean(samples),
        }
        record["cases"].append(row)
        save()
        print(json.dumps({"plan_index": index, "latency_ms": row["latency_ms"], "cv": row["cv"]}), flush=True)

    save()
    try:
        for index, case in selected:
            collect(index, case)
            gc.collect()
            torch.npu.empty_cache()
        record["status"] = "completed"
    except Exception:
        record.update(status="failed", failed_plan_index=index, error=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
