"""Collect HCCL curves with explicit placement, dtype and execution identity.

Run with torchrun inside the pinned Ascend runtime. AllReduce uses zero payloads
during repeated timing to avoid in-place overflow; signed rank/index patterns
are checked over the full output before and after timing for every operation.
These are raw observations pending producer/consumer qualification.
"""

import argparse
import datetime
import gc
import json
import os
import socket
import statistics
import time
from pathlib import Path


def summarize(samples):
    average = statistics.mean(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "cv": statistics.pstdev(samples) / average,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks-per-node", type=int, required=True)
    parser.add_argument("--sizes", default="256,4096,12288,49152,196608,786432,3145728,12582912,50331648,100663296")
    parser.add_argument("--dtypes", default="bfloat16")
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--operations", default="all_reduce,all_gather,reduce_scatter,all_to_all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--gc-policy", choices=("enabled", "disabled"), default="enabled")
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    dtypes = args.dtypes.split(",")
    modes = args.modes.split(",")
    operations = args.operations.split(",")
    if min(*sizes, args.ranks_per_node, args.warmup, args.repeats, args.iterations) <= 0:
        parser.error("sizes and iteration counts must be positive")
    if not set(dtypes) <= {"bfloat16", "float16", "float32", "int32"}:
        parser.error("unsupported dtype")
    if not set(modes) <= {"eager", "graph"}:
        parser.error("unsupported execution mode")
    if not set(operations) <= {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}:
        parser.error("unsupported collective")

    import torch
    import torch.distributed as distributed
    import torch_npu

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if int(os.environ["LOCAL_WORLD_SIZE"]) != args.ranks_per_node or world % args.ranks_per_node:
        raise ValueError("torchrun placement does not match ranks-per-node")
    torch.npu.set_device(local_rank)
    distributed.init_process_group("hccl", timeout=datetime.timedelta(seconds=300))
    control = distributed.new_group(backend="gloo", timeout=datetime.timedelta(seconds=300))
    identity = {
        "host": socket.gethostname(), "rank": rank, "local_rank": local_rank,
        "device": torch.npu.get_device_name(local_rank), "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
    }
    identities = [None] * world
    distributed.all_gather_object(identities, identity, group=control)
    result = {
        "schema": "aisimulate-hccl-observations/v1", "status": "running",
        "num_ranks": world, "ranks_per_node": args.ranks_per_node, "rank_layout": "contiguous",
        "devices": identities, "warmup": args.warmup, "repeats": args.repeats,
        "iterations": args.iterations, "canonical_fpm_data": False, "cases": [],
        "gc_policy": args.gc_policy,
    }

    def save():
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")

    def barrier():
        distributed.barrier(group=control)

    def collect(operation, dtype_name, size_bytes, mode):
        dtype = getattr(torch, dtype_name)
        element_size = torch.empty((), dtype=dtype).element_size()
        if size_bytes % element_size or (
            operation in {"reduce_scatter", "all_to_all"} and size_bytes % (world * element_size)
        ):
            raise ValueError("Payload is not aligned to the collective element/partition size")
        count = size_bytes // element_size
        source = torch.zeros(count, dtype=dtype, device="npu")
        if operation == "all_reduce":
            target = source
            invoke = lambda: distributed.all_reduce(target)
        elif operation == "all_gather":
            target = torch.empty(count * world, dtype=dtype, device="npu")
            invoke = lambda: distributed.all_gather_into_tensor(target, source)
        elif operation == "reduce_scatter":
            target = torch.empty(count // world, dtype=dtype, device="npu")
            invoke = lambda: distributed.reduce_scatter_tensor(target, source)
        else:
            target = torch.empty_like(source)
            invoke = lambda: distributed.all_to_all_single(target, source)

        def verify():
            pattern = (torch.arange(count, dtype=torch.int32, device="npu") % 5 - 2).to(dtype)
            source.copy_(pattern + (rank % 7 - 3))
            invoke()
            torch.npu.synchronize()
            rank_sum = sum(peer % 7 - 3 for peer in range(world))
            if operation == "all_reduce":
                passed = torch.equal(target, pattern * world + rank_sum)
            elif operation == "reduce_scatter":
                portion = pattern[rank * (count // world) : (rank + 1) * (count // world)]
                passed = torch.equal(target, portion * world + rank_sum)
            else:
                portion = pattern if operation == "all_gather" else pattern[
                    rank * (count // world) : (rank + 1) * (count // world)
                ]
                passed = all(
                    torch.equal(target.reshape(world, -1)[peer], portion + (peer % 7 - 3))
                    for peer in range(world)
                )
            states = [None] * world
            distributed.all_gather_object(states, passed, group=control)
            if not all(states):
                raise ValueError(f"Full-payload correctness failure on ranks {[index for index, state in enumerate(states) if not state]}")
            source.zero_()
            torch.npu.synchronize()

        verify()
        for _ in range(args.warmup):
            invoke()
        torch.npu.synchronize()
        graph = None
        if mode == "graph":
            graph = torch.npu.NPUGraph()
            barrier()
            with torch.npu.graph(graph):
                for _ in range(args.iterations):
                    invoke()
            for _ in range(args.warmup):
                graph.replay()
            torch.npu.synchronize()
        samples = []
        wall_samples = []
        gc_samples = []
        gc_state = {"start": 0.0, "duration": 0.0}

        def observe_gc(phase, info):
            if phase == "start":
                gc_state["start"] = time.perf_counter()
            else:
                gc_state["duration"] += time.perf_counter() - gc_state["start"]

        gc_was_enabled = gc.isenabled()
        if args.gc_policy == "disabled":
            gc.collect()
            gc.disable()
        gc.callbacks.append(observe_gc)
        try:
            for _ in range(args.repeats):
                barrier()
                start_event = torch.npu.Event(enable_timing=True)
                end_event = torch.npu.Event(enable_timing=True)
                gc_state["duration"] = 0.0
                start_event.record()
                started = time.perf_counter()
                if graph is not None:
                    graph.replay()
                else:
                    for _ in range(args.iterations):
                        invoke()
                end_event.record()
                end_event.synchronize()
                wall_samples.append((time.perf_counter() - started) * 1000 / args.iterations)
                samples.append(start_event.elapsed_time(end_event) / args.iterations)
                gc_samples.append(gc_state["duration"] * 1000)
        finally:
            gc.callbacks.remove(observe_gc)
            if gc_was_enabled:
                gc.enable()
        verify()
        rank_samples = [None] * world
        distributed.all_gather_object(rank_samples, {"device_ms": samples, "wall_ms": wall_samples, "gc_ms": gc_samples}, group=control)
        critical = [max(values["device_ms"][index] for values in rank_samples) for index in range(args.repeats)]
        row = {
            "operation": operation, "dtype": dtype_name, "input_bytes_per_rank": size_bytes,
            "execution_mode": mode, "rank_samples": rank_samples, "critical_device": summarize(critical),
            "full_payload_correctness_checks": 2 * world,
            "timing_scope": "NPU events over collective calls; eager includes device idle gaps from host submission",
        }
        row["repeatable_at_10_percent"] = row["critical_device"]["cv"] <= 0.10
        return row

    save()
    try:
        for dtype in dtypes:
            for operation in operations:
                for mode in modes:
                    for size in sizes:
                        identity = {"dtype": dtype, "operation": operation, "execution_mode": mode, "input_bytes_per_rank": size}
                        try:
                            row = collect(operation, dtype, size, mode)
                        except Exception as error:
                            result["failed_case"] = {**identity, "error": f"{type(error).__name__}: {error}"}
                            raise
                        result["cases"].append(row)
                        save()
                        if rank == 0:
                            print(json.dumps({**identity, **row["critical_device"]}), flush=True)
                        torch.npu.empty_cache()
        result["status"] = "completed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
