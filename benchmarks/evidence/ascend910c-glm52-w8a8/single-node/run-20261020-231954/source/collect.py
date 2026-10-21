import argparse
import datetime
import importlib.metadata
import json
import math
import os
import pathlib
import statistics
import time
import traceback

import torch
import torch.distributed as distributed
import torch_npu


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def distribution(samples):
    average = statistics.mean(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "cv": statistics.stdev(samples) / average if len(samples) > 1 else 0,
    }


def measure(operation, args):
    for _ in range(args.warmup):
        operation()
    torch.npu.synchronize()
    event_samples = []
    wall_samples = []
    for _ in range(args.repeats):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        for _ in range(args.iterations):
            operation()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - started) * 1000 / args.iterations)
        event_samples.append(begin.elapsed_time(end) / args.iterations)
    if not all(math.isfinite(value) and value > 0 for value in event_samples):
        raise ValueError("Invalid device timing samples")
    return {"device": distribution(event_samples), "wall": distribution(wall_samples)}


def memory_case(device, size_bytes, args):
    source = torch.full((size_bytes,), 37, dtype=torch.uint8, device=device)
    target = torch.empty_like(source)
    operation = lambda: target.copy_(source)
    operation()
    if not torch.equal(target.cpu(), source.cpu()):
        raise AssertionError("HBM copy correctness failed")
    timing = measure(operation, args)
    return {
        "operation": "hbm_copy",
        "kernel_source": "torch.Tensor.copy_",
        "device": device.index,
        "size_bytes": size_bytes,
        "traffic_bytes": 2 * size_bytes,
        "traffic_definition": "one read plus one write; not physical counter traffic",
        "effective_bytes_per_second": 2 * size_bytes / (timing["device"]["median_ms"] / 1000),
        "correctness": "full buffer byte equality",
        "timing": timing,
    }


def matmul_case(device, rows, inner, columns, precision, args):
    activation_int = torch.randint(-2, 3, (rows, inner), dtype=torch.int8, device=device)
    weight_int = torch.randint(-2, 3, (columns, inner), dtype=torch.int8, device=device)
    if precision == "bf16":
        activation = activation_int.to(torch.bfloat16)
        weight = weight_int.to(torch.bfloat16).t()
        output = torch.empty((rows, columns), device=device, dtype=torch.bfloat16)
        operation = lambda: torch.mm(activation, weight, out=output)
        kernel_source = "torch.mm"
    else:
        activation = activation_int
        weight = weight_int.t()
        scale = torch.ones(columns, device=device, dtype=torch.bfloat16)
        operation = lambda: torch_npu.npu_quant_matmul(
            activation, weight, scale, output_dtype=torch.bfloat16
        )
        kernel_source = "torch_npu.npu_quant_matmul"
    result = operation()
    row_indices = sorted({0, rows // 2, rows - 1})
    column_indices = sorted({0, columns // 2, columns - 1})
    expected = activation_int[row_indices].cpu().float() @ weight_int[column_indices].cpu().float().t()
    actual = result[row_indices][:, column_indices].cpu().float()
    torch.testing.assert_close(actual, expected.to(torch.bfloat16).float(), rtol=0.008, atol=1)
    timing = measure(operation, args)
    return {
        "operation": "matmul",
        "kernel_source": kernel_source,
        "device": device.index,
        "precision": precision,
        "output_dtype": "bfloat16",
        "shape_mkn": [rows, inner, columns],
        "weight_layout": "ND transposed view of [N,K]",
        "quantization": "unit per-channel scale, no activation quantization" if precision == "int8" else None,
        "arithmetic_operations": 2 * rows * inner * columns,
        "effective_ops_per_second": 2 * rows * inner * columns / (timing["device"]["median_ms"] / 1000),
        "correctness": "up to nine output positions versus independent CPU FP32 dot products",
        "timing": timing,
    }


def single_device(args, report):
    report["devices"] = []
    for device_id in args.devices:
        device = torch.device("npu", device_id)
        torch.npu.set_device(device)
        torch.npu.manual_seed(20261020 + device_id)
        free_bytes, total_bytes = torch.npu.mem_get_info(device)
        report["devices"].append({
            "id": device_id,
            "name": torch.npu.get_device_name(device_id),
            "free_memory_bytes_before_cases": free_bytes,
            "total_memory_bytes": total_bytes,
        })
        cases = [("memory", size, 0, 0) for size in [128 << 20, 512 << 20]]
        cases += [(precision, 4096, 4096, 4096) for precision in ["bf16", "int8"]]
        if device_id == args.devices[0]:
            cases += [("memory", 1024 << 20, 0, 0)]
            cases += [(precision, 8192, 8192, 8192) for precision in ["bf16", "int8"]]
            cases += [
                (precision, rows, 6144, 2048)
                for precision in ["bf16", "int8"]
                for rows in [1, 8, 32, 128, 512, 2048, 8192]
            ]
        for precision, rows, inner, columns in cases:
            identity = {"device": device_id, "precision": precision, "shape": [rows, inner, columns]}
            print(json.dumps({"starting": identity}), flush=True)
            try:
                if precision == "memory":
                    result = memory_case(device, rows, args)
                else:
                    result = matmul_case(device, rows, inner, columns, precision, args)
                report["cases"].append(result)
            except Exception as error:
                report["errors"].append({"case": identity, "error": repr(error), "traceback": traceback.format_exc()})
                write_json(args.output, report)
                raise
            write_json(args.output, report)
        torch.npu.empty_cache()


def collective_operation(name, size_bytes, rank, world, device):
    count = size_bytes // 2
    source = torch.full((count,), rank + 1, dtype=torch.bfloat16, device=device)
    if name == "all_reduce":
        output = source
        operation = lambda: distributed.all_reduce(source)
        expected = world * (world + 1) // 2
    elif name == "all_gather":
        output = torch.empty(count * world, dtype=source.dtype, device=device)
        operation = lambda: distributed.all_gather_into_tensor(output, source)
        expected = torch.arange(1, world + 1, dtype=torch.float32).repeat_interleave(count)
    elif name == "reduce_scatter":
        output = torch.empty(count // world, dtype=source.dtype, device=device)
        operation = lambda: distributed.reduce_scatter_tensor(output, source)
        expected = world * (world + 1) // 2
    else:
        output = torch.empty_like(source)
        operation = lambda: distributed.all_to_all_single(output, source)
        expected = torch.arange(1, world + 1, dtype=torch.float32).repeat_interleave(count // world)
    operation()
    torch.npu.synchronize()
    actual = output.cpu().float()
    if isinstance(expected, int):
        if not bool(torch.all(actual == expected)):
            raise AssertionError("Collective sum validation failed")
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    source.zero_()
    torch.npu.synchronize()
    return operation


def collectives(args, report):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("npu", args.devices[rank])
    torch.npu.set_device(device)
    distributed.init_process_group("hccl", timeout=datetime.timedelta(seconds=180))
    try:
        for name in ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]:
            for size_bytes in [4096, 65536, 1 << 20, 16 << 20, 64 << 20]:
                identity = {"operation": name, "size_bytes": size_bytes, "devices": args.devices}
                if rank == 0:
                    print(json.dumps({"starting": identity}), flush=True)
                operation = collective_operation(name, size_bytes, rank, world, device)
                for _ in range(args.warmup):
                    operation()
                torch.npu.synchronize()
                rank_samples = []
                for _ in range(args.repeats):
                    distributed.barrier()
                    torch.npu.synchronize()
                    started = time.perf_counter()
                    for _ in range(args.iterations):
                        operation()
                    torch.npu.synchronize()
                    rank_samples.append((time.perf_counter() - started) * 1000 / args.iterations)
                samples_tensor = torch.tensor(rank_samples, dtype=torch.float32, device=device)
                gathered = [torch.empty_like(samples_tensor) for _ in range(world)]
                distributed.all_gather(gathered, samples_tensor)
                if rank == 0:
                    per_rank = [sample.cpu().tolist() for sample in gathered]
                    critical_samples = [max(sample[index] for sample in per_rank) for index in range(args.repeats)]
                    timing = distribution(critical_samples)
                    report["cases"].append({
                        **identity,
                        "kernel_source": "torch.distributed." + name + " (ProcessGroupHCCL)",
                        "dtype": "bfloat16",
                        "payload_definition": "input bytes per rank; all_gather output is world times input",
                        "measurement": "maximum rank wall time per repeat; barrier excluded; eager host submission included",
                        "rank_samples_ms": per_rank,
                        "timing": timing,
                        "input_bytes_per_second": size_bytes / (timing["median_ms"] / 1000),
                        "correctness": "full output equality to rank-specific collective reference before warmup",
                    })
                    write_json(args.output, report)
                del operation
                torch.npu.empty_cache()
    except Exception as error:
        if rank == 0:
            report["errors"].append({"error": repr(error), "traceback": traceback.format_exc()})
            write_json(args.output, report)
        raise
    finally:
        distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["single", "communication"])
    parser.add_argument("--devices", type=lambda value: [int(item) for item in value.split(",")], required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) <= 0 or len(set(args.devices)) != len(args.devices):
        parser.error("Counts must be positive and devices must be unique")
    if args.output.exists():
        parser.error("Output already exists; use a fresh output file")
    torch.set_num_threads(1)
    report = {
        "schema": "ascend-single-node-calibration/v1",
        "scope": "hardware microbenchmark; not a serving operator profile or theoretical peak",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": args.mode,
        "parameters": {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()},
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "torch-npu", "sglang"]},
        "cases": [],
        "errors": [],
        "complete": False,
    }
    with torch.inference_mode():
        if args.mode == "single":
            single_device(args, report)
        else:
            collectives(args, report)
    if int(os.environ.get("RANK", "0")) == 0:
        report["complete"] = True
        report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(args.output, report)


if __name__ == "__main__":
    main()
