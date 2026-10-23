"""Inspect source-selected DeepEP transport outputs before writing timing cases."""

import argparse
import datetime
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2)
    args = parser.parse_args()
    setup = json.loads(args.setup.read_text())
    for name, value in setup["environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    local_rank = int(os.environ["LOCAL_RANK"])
    os.sched_setaffinity(0, setup["worker_cpu_affinity"][str(local_rank)])
    import torch
    import torch.distributed as distributed
    import torch_npu

    from sglang.srt.distributed.parallel_state import get_torch_distributed_pg_options
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPBuffer,
        _DeepEPDispatcherImplLowLatency,
        _DeepEPDispatcherImplNormal,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import DeepEPMode
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    init_npu_backend()
    torch.npu.set_device(local_rank)
    torch.set_num_threads(4)
    distributed.init_process_group("hccl", timeout=datetime.timedelta(seconds=300), pg_options=get_torch_distributed_pg_options())
    control = distributed.new_group(backend="gloo", timeout=datetime.timedelta(seconds=300))
    rank, world = distributed.get_rank(), distributed.get_world_size()
    publish(ServerArgs(model_path="/campaign/model-metadata", device="npu", tp_size=world,
                       ep_size=world, dp_size=world // 8, enable_dp_attention=True,
                       enable_dp_lm_head=True, moe_a2a_backend="deepep", quantization="modelslim",
                       dtype="bfloat16", context_length=180000, disable_shared_experts_fusion=True), role="scheduler")
    common = dict(group=distributed.group.WORLD, router_topk=8, permute_fusion=False, num_experts=256, num_local_experts=256 // world, hidden_size=6144, params_dtype=torch.bfloat16, deepep_mode=DeepEPMode.AUTO)
    normal = _DeepEPDispatcherImplNormal(async_finish=True, **common)
    low_latency = _DeepEPDispatcherImplLowLatency(return_recv_hook=True, **common)
    for dispatcher in (normal, low_latency):
        dispatcher.set_quant_config({"dispatcher_output_dtype": "int8"})
    global_ids = torch.arange(args.tokens, device="npu", dtype=torch.int64) + rank * args.tokens
    columns = torch.arange(6144, device="npu", dtype=torch.int64)
    values = ((global_ids[:, None] * 13 + columns[None, :]) % 15 - 7).to(torch.bfloat16)
    values[:, 0] = (global_ids % 127).to(torch.bfloat16)
    values[:, 1] = (global_ids // 127).to(torch.bfloat16)
    values[:, -1] = 127
    ids = (global_ids[:, None] * 8 + torch.arange(8, device="npu", dtype=torch.int64)) % 256
    weights = torch.full((args.tokens, 8), 0.125, device="npu", dtype=torch.float32)
    topk = StandardTopKOutput(weights, ids.to(torch.int32), torch.empty(args.tokens, 256, device="npu", dtype=torch.bfloat16))

    def describe(value):
        if isinstance(value, torch.Tensor):
            result = {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype), "format": torch_npu.get_npu_format(value)}
            if value.numel() <= 128:
                result["values"] = value.cpu().tolist()
            return result
        if isinstance(value, (tuple, list)):
            return [describe(item) for item in value]
        if value is None or isinstance(value, (int, float, str, bool)):
            return value
        return type(value).__module__ + "." + type(value).__name__

    observations = []
    for phase, dispatcher in (("normal", normal), ("low_latency", low_latency)):
        distributed.barrier(group=control)
        dispatched = dispatcher.dispatch_b(*dispatcher.dispatch_a(values, topk))
        torch.npu.synchronize()
        hidden, scales = dispatched.hidden_states, dispatched.hidden_states_scale
        if hidden.dtype != torch.int8 or scales is None:
            raise ValueError("Actual dispatcher did not select INT8 token quantization")
        sample = hidden[: min(hidden.shape[0], 8), :4].cpu().tolist()
        handle = describe(dispatcher.handle)
        restored = (hidden.float() * scales.reshape(-1, 1)).to(torch.bfloat16)
        if phase == "normal":
            combined = dispatcher.combine_b(*dispatcher.combine_a(restored, dispatched.topk_ids, dispatched.topk_weights))
        else:
            combined = dispatcher.combine_b(*dispatcher.combine_a(restored, ids, weights))
        torch.npu.synchronize()
        error = (combined.float() - values.float()).abs()
        observations.append({"phase": phase, "dispatch": describe(dispatched), "handle": handle, "sample_first_columns": sample, "combine_shape": list(combined.shape), "identity_roundtrip_max_error": error.max().item(), "input_sample": values[:2, :4].cpu().tolist(), "output_sample": combined[:2, :4].cpu().tolist(), "buffer_strategies": {"normal": type(dispatcher._get_buffer().normal_strategy).__name__, "low_latency": type(dispatcher._get_buffer().low_latency_strategy).__name__}})
    distributed.barrier(group=control)
    dispatch_graph = torch.npu.NPUGraph()
    with torch.npu.graph(dispatch_graph):
        graph_dispatch = low_latency.dispatch_b(*low_latency.dispatch_a(values, topk))
    saved_handle, saved_counts = low_latency.handle, low_latency.packed_recv_count
    dispatch_graph.replay()
    torch.npu.synchronize()
    values.neg_()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    begin.record()
    dispatch_graph.replay()
    end.record()
    end.synchronize()
    dispatch_ms = begin.elapsed_time(end)
    combine_input = (graph_dispatch.hidden_states.float() * graph_dispatch.hidden_states_scale.reshape(-1, 1)).to(torch.bfloat16)
    low_latency.handle, low_latency.packed_recv_count = saved_handle, saved_counts
    combined = low_latency.combine_b(*low_latency.combine_a(combine_input, ids, weights))
    torch.npu.synchronize()
    if not torch.equal(combined, values):
        raise ValueError("Actual dispatch graph replay failed the full-payload roundtrip")
    combine_graph = torch.npu.NPUGraph()
    low_latency.handle, low_latency.packed_recv_count = saved_handle, saved_counts
    with torch.npu.graph(combine_graph):
        graph_combined = low_latency.combine_b(*low_latency.combine_a(combine_input, ids, weights))
    combine_graph.replay()
    torch.npu.synchronize()
    combine_input.neg_()
    values.neg_()
    begin.record()
    combine_graph.replay()
    end.record()
    end.synchronize()
    if not torch.equal(graph_combined, values):
        raise ValueError("Actual combine graph replay failed the full-payload roundtrip")
    observations.append({"phase": "low_latency_graph", "identity_roundtrip_max_error": 0.0,
                         "dispatch_ms": dispatch_ms,
                         "combine_ms": begin.elapsed_time(end),
                         "scope": "Separate captured graphs with outer events and actual output mutation checks; not qualified repeated timings"})
    gathered = [None] * world
    distributed.all_gather_object(gathered, {"rank": rank, "observations": observations}, group=control)
    if rank == 0:
        args.output.write_text(json.dumps({"status": "observed", "scope": "Transport contract probe; normal combine may need runner permutation/weighting before a roundtrip check", "world": world, "tokens_per_rank": args.tokens, "ranks": gathered}, indent=2) + "\n")
        print(json.dumps({"status": "observed", "world": world, "max_roundtrip_error": max(entry["identity_roundtrip_max_error"] for row in gathered for entry in row["observations"])}))
    distributed.barrier(group=control)
    DeepEPBuffer.clean_buffer()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
