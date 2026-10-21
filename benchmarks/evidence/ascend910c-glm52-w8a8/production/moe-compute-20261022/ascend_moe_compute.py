"""Measure observed Ascend expert-count cases and weight-cache controls.

This experimental collector does not publish production prediction rows. Counts
and storage geometry come from serving; tensor values are seeded controls.

Serving contract at SGLang baa9befa381511cd4d56f83980951e5a45fffc6d:
hardware_backend/npu/quantization/moe_methods.py:648-711 prepares NZ INT8
weights, BF16 channel scales and the W8A8 grouped matmul. Its matmul.py:41
passes count-form group lists unchanged. layers/moe/moe_runner/ascend.py:139-169
selects NPUSwigluDeepEPKernel for this DeepEP W8A8 configuration; :234-275
passes GMM1 output through that activation to GMM2 with the same group list.
These are calls to installed framework implementations, not copied kernels.
"""

import argparse
import gc
import gzip
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


def validate_case(case):
    counts = case["expert_counts"]
    rows, hidden = case["input_shape"]
    experts, weight_hidden, doubled_intermediate = case["weight_shape"]
    if (
        case["input_dtype"] != "torch.int8"
        or case["weight_dtype"] != "torch.int8"
        or case["output_dtype"] != "torch.bfloat16"
        or case["group_list_type"] != 1
        or case["execution"] not in {"eager", "graph"}
        or any(type(value) is not int or value <= 0 for value in (rows, hidden, experts, doubled_intermediate))
        or hidden != weight_hidden
        or doubled_intermediate % 2
        or len(counts) != experts
        or any(type(value) is not int or value < 0 for value in counts)
        or sum(counts) > rows
    ):
        raise ValueError("Unsupported or inconsistent captured MoE geometry")
    return rows, hidden, doubled_intermediate // 2, experts


def selected_rows(counts):
    result = []
    start = 0
    for expert, count in enumerate(counts):
        if count:
            result.extend((expert, row) for row in sorted({start, start + count - 1}))
        start += count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--serving-setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--weight-copies", default="1,8")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--graph-calls", type=int, default=16)
    args = parser.parse_args()
    copies = [int(value) for value in args.weight_copies.split(",")]
    if min(*copies, args.warmup, args.repeats, args.graph_calls) <= 0:
        parser.error("measurement counts must be positive")
    if any(args.graph_calls % count for count in copies) or len(set(copies)) != len(copies):
        parser.error("weight copies must be distinct divisors of graph-calls")
    if args.output.exists():
        parser.error("output must be fresh; preserve previous attempts")
    setup = json.loads(args.serving_setup.read_text())
    if setup["schema"] != "sglang-runtime-setup/v1":
        raise ValueError("Unsupported serving setup")
    for name, value in setup["environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    affinity = setup["worker_cpu_affinity"][str(args.device)]
    if not affinity or not set(affinity) <= os.sched_getaffinity(0):
        raise ValueError("Recorded serving CPU mask is unavailable")
    os.sched_setaffinity(0, affinity)
    opener = gzip.open if args.cases.suffix == ".gz" else open
    with opener(args.cases, "rt") as stream:
        cases = json.load(stream)
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Case list must be nonempty with unique identities")
    for case in cases:
        validate_case(case)

    import torch
    import torch_npu

    importlib.import_module("sglang.srt.layers.quantization")
    from sglang.srt.hardware_backend.npu.moe.activation import NPUSwigluDeepEPKernel
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend, npu_format_cast

    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    if revision != "baa9befa381511cd4d56f83980951e5a45fffc6d":
        raise RuntimeError("Reaudit dispatch and metadata before using another runtime")
    torch.set_num_threads(8)
    init_npu_backend()
    torch.npu.set_device(args.device)
    torch.manual_seed(910)
    kernel = NPUW8A8Int8MoEMethod()
    activation = NPUSwigluDeepEPKernel(need_quant=True)
    result = {
        "schema": "aisimulate-ascend-moe-compute/v1",
        "status": "running",
        "canonical_fpm_data": False,
        "input_values": "Seeded INT8 [-3,3], per-token scale 1/32, BF16 channel scale 1/8; not checkpoint values",
        "case_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {
            "sglang_commit": revision,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "host": socket.gethostname(),
            "device": torch.npu.get_device_name(args.device),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "serving_setup": setup,
        },
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "graph_calls": args.graph_calls,
            "weight_copies": copies,
            "clock": "Outer NPU events and host wall clock around graph replay; both divided by graph calls",
            "scope": "Standalone graph compute with dependencies; excludes dispatch/combine and serving overlap",
            "cache_control": "Rotate distinct NZ weights; no cold-cache claim without convergence evidence",
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
            raise ValueError(f"Sampled numerical reference mismatch: max={difference.max().item()}")
        return {"elements": actual.numel(), "max_abs_error": difference.max().item()}

    def measure(functions):
        for _ in range(args.warmup):
            for function in functions:
                function()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for index in range(args.graph_calls):
                function = functions[index % len(functions)]
                function()
        for _ in range(args.warmup):
            graph.replay()
        torch.npu.synchronize()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        samples, wall_samples = [], []
        for _ in range(args.repeats):
            started = time.perf_counter()
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / args.graph_calls)
            wall_samples.append((time.perf_counter() - started) * 1000 / args.graph_calls)
        return {
            "samples_ms": samples,
            "wall_samples_ms": wall_samples,
            "median_ms": statistics.median(samples),
            "cv": statistics.pstdev(samples) / statistics.mean(samples),
        }

    def collect(case):
        rows, hidden, intermediate, experts = validate_case(case)
        counts = torch.tensor(case["expert_counts"], device="npu", dtype=torch.int64)
        inputs = torch.randint(-3, 4, (rows, hidden), device="npu", dtype=torch.int8)
        scales = torch.full((rows,), 1 / 32, device="npu", dtype=torch.float32)
        weight13 = torch.randint(-3, 4, (experts, hidden, 2 * intermediate), device="npu", dtype=torch.int8)
        weight2 = torch.randint(-3, 4, (experts, intermediate, hidden), device="npu", dtype=torch.int8)
        weights = []
        for _ in range(max(copies)):
            weights.append(
                SimpleNamespace(
                    w13_weight=npu_format_cast(weight13.clone()),
                    w2_weight=npu_format_cast(weight2.clone()),
                    w13_weight_scale=torch.full((experts, 2 * intermediate), 0.125, device="npu", dtype=torch.bfloat16),
                    w2_weight_scale=torch.full((experts, hidden), 0.125, device="npu", dtype=torch.bfloat16),
                )
            )

        def gmm1(weight):
            return kernel.apply(weight, inputs, counts, scales, torch.bfloat16, "w13", 1)

        intermediate_values = gmm1(weights[0])
        quantized, intermediate_scales = activation._apply_activation(intermediate_values, counts, 1)

        def gmm2(weight):
            return kernel.apply(weight, quantized, counts, intermediate_scales, torch.bfloat16, "w2", 1)

        def chain(weight):
            gate_up = gmm1(weight)
            quantized_values, token_scales = activation._apply_activation(gate_up, counts, 1)
            return kernel.apply(weight, quantized_values, counts, token_scales, torch.bfloat16, "w2", 1)

        output = gmm2(weights[0])
        checks = []
        for expert, row in selected_rows(case["expert_counts"]):
            columns = list(range(0, intermediate * 2, max(1, intermediate * 2 // 16)))
            expected = inputs[row].float().cpu() @ weight13[expert, :, columns].float().cpu() / 256
            checks.append(compare(intermediate_values[row, columns], expected))
            original = intermediate_values[row].float().cpu()
            activated = torch.nn.functional.silu(original[:intermediate]) * original[intermediate:]
            expected_scale = activated.abs().max() / 127
            observed_scale = intermediate_scales[row].float().cpu()
            torch.testing.assert_close(observed_scale, expected_scale, rtol=2e-4, atol=1e-7)
            expected_quant = (activated / expected_scale).round().clamp(-128, 127)
            if (quantized[row].float().cpu() - expected_quant).abs().max().item() > 1:
                raise ValueError("Fused activation differs by more than one quantization bin")
            columns = list(range(0, hidden, max(1, hidden // 16)))
            expected = quantized[row].float().cpu() @ weight2[expert, :, columns].float().cpu()
            expected *= observed_scale * 0.125
            checks.append(compare(output[row, columns], expected))
        entry = {
            "case": case,
            "kernel_sources": {
                "gmm": type(kernel.matmul).__module__ + "." + type(kernel.matmul).__name__,
                "activation": activation._kernel.__module__ + "." + activation._kernel.__name__,
            },
            "correctness": {
                "passed": True,
                "sampled_rows": len(selected_rows(case["expert_counts"])),
                "checks": checks,
            },
            "observed_intermediate_shapes": [
                list(intermediate_values.shape),
                list(quantized.shape),
                list(output.shape),
            ],
            "measurements": [],
        }
        result["cases"].append(entry)
        del weight13, weight2
        for count in copies:
            for operation in ("gmm1", "activation", "gmm2", "chain"):
                functions = []
                for weight in weights[:count]:
                    if operation == "gmm1":
                        functions.append(lambda weight=weight: gmm1(weight))
                    elif operation == "gmm2":
                        functions.append(lambda weight=weight: gmm2(weight))
                    elif operation == "chain":
                        functions.append(lambda weight=weight: chain(weight))
                    else:
                        functions.append(lambda: activation._apply_activation(intermediate_values, counts, 1))
                measurement = {"operation": operation, "weight_copies": count, **measure(functions)}
                entry["measurements"].append(measurement)
                save()
                print(json.dumps({"case_id": case["case_id"], **measurement}), flush=True)

    save()
    try:
        for case in cases:
            collect(case)
            gc.collect()
            torch.npu.empty_cache()
        result["status"] = "completed"
    except Exception:
        result["status"] = "failed"
        result["failed_case_id"] = case["case_id"]
        result["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
