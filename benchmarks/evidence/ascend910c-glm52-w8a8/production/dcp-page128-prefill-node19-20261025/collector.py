"""Measure the installed NPU DCP attention chain across its real HCCL group.

Calls SGLang baa9befa381511cd4d56f83980951e5a45fffc6d's sparse-index
remapper, query gather, partial sparse attention and NPU output merger.
The last helper uses FP32 all-to-all and npu_attention_update. No DeepEP,
indexer, Q/KV projections or absorbed BMM is included in this boundary.
"""

import argparse
import datetime
import gc
import hashlib
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
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous observations; use a fresh output")
    plan, setup = json.loads(args.plan.read_text()), json.loads(args.serving_setup.read_text())
    protocol = plan["protocol"]
    if (
        plan["schema"] != "aisimulate-ascend-dcp-attention-plan/v1"
        or min(protocol.values()) < 1
        or protocol["repeats"] < 3
    ):
        raise ValueError("Invalid DCP plan or measurement protocol")
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
    from sglang.srt.distributed.parallel_state import get_torch_distributed_pg_options, init_model_parallel_group
    from sglang.srt.hardware_backend.npu.attention.dsa_dcp import forward_dcp_sparse_attention
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    from sglang.srt.layers.dcp.comm import all_gather_q_for_mla_decode, cp_lse_ag_out_rs_mla_npu
    from sglang.srt.layers.dcp.layout import remap_dcp_sparse_indices
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.runtime_context import get_parallel, publish
    from sglang.srt.server_args import ServerArgs

    init_npu_backend()
    torch.npu.set_device(local_rank)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    if revision != "baa9befa381511cd4d56f83980951e5a45fffc6d":
        raise ValueError("Reaudit DCP dispatch for another runtime")
    distributed.init_process_group(
        "hccl", timeout=datetime.timedelta(seconds=300), pg_options=get_torch_distributed_pg_options()
    )
    rank, world = distributed.get_rank(), distributed.get_world_size()
    if int(os.environ["LOCAL_WORLD_SIZE"]) != world or world not in (2, 4, 8, 16):
        raise ValueError("The qualified DCP group is contiguous within one node")
    publish(
        ServerArgs(
            model_path=str(args.model_metadata),
            device="npu",
            tp_size=world,
            dcp_size=world,
            attn_dp_size=1,
            ep_size=1,
            moe_a2a_backend="none",
            quantization="modelslim",
            dtype="bfloat16",
            attention_backend="ascend",
            context_length=180000,
            disable_shared_experts_fusion=True,
        ),
        role="scheduler",
    )
    group = init_model_parallel_group(
        [list(range(world))],
        local_rank,
        "hccl",
        use_pynccl=False,
        use_custom_allreduce=False,
        use_mscclpp=False,
        use_torch_symm_mem_allreduce=False,
        group_name="dcp",
    )
    get_parallel().override_permanently(dcp_group=group)
    selected = [
        (index, case)
        for index, case in enumerate(plan["cases"])
        if case["dcp_size"] == world and (args.case_index is None or index == args.case_index)
    ]
    if not selected or (args.negative_control and (len(selected) != 1 or selected[0][1]["execution_mode"] != "graph")):
        raise ValueError("Empty DCP selection or invalid negative control")
    identities = [None] * world
    distributed.all_gather_object(
        identities,
        {
            "rank": rank,
            "local_rank": local_rank,
            "host": socket.gethostname(),
            "device": torch.npu.get_device_name(local_rank),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "driver_version_info": Path("/usr/local/Ascend/driver/version.info").read_text(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        group=group.cpu_group,
    )
    record = {
        "schema": "aisimulate-ascend-dcp-attention-observations/v1",
        "status": "running",
        "scope": plan["scope"],
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "framework_commit": revision,
        "runtime": identities,
        "serving_setup": setup,
        "protocol": protocol,
        "selected_plan_indices": [index for index, _ in selected],
        "cases": [],
        "negative_control": args.negative_control,
    }

    def save():
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2) + "\n")

    def collect(index, case):
        tokens, active, padded = case["tokens"], case["active_batch"], case["padded_batch"]
        heads, length, prefix = case["local_heads"], case["sequence_length"], case["prefix"]
        query_per_request, page = case["query_tokens_per_request"], case["page_size"]
        if (
            case["kv_interleave_size"] != page
            or (case["kv_lora_rank"], case["rope_dim"], page, case["sparse_topk"]) != (512, 64, 128, 2048)
            or heads not in (4, 8, 16)
            or heads * world > 64
            or (
                case["phase"] == "prefill"
                and (tokens != active * query_per_request or length != prefix + query_per_request)
            )
            or (case["phase"] == "decode" and (tokens != padded or query_per_request != 1))
        ):
            raise ValueError("Invalid source-qualified DCP geometry")
        global_generator = torch.Generator(device="cpu").manual_seed(910 + index)
        query_generator = torch.Generator(device="cpu").manual_seed(1910 + index * world + rank)
        global_keys = torch.randn((active, length, 512), generator=global_generator).bfloat16()
        global_rope = torch.randn((active, length, 64), generator=global_generator).bfloat16()
        query_cpu = torch.randn((tokens, heads, 512), generator=query_generator).bfloat16()
        rope_cpu = torch.randn((tokens, heads, 64), generator=query_generator).bfloat16()
        query, query_rope = query_cpu.npu(), rope_cpu.npu()
        global_positions = torch.arange(length, dtype=torch.int64)
        owned_positions = global_positions[(global_positions // page) % world == rank]
        local_length = owned_positions.numel()
        pages_per_request = (length + page * world - 1) // (page * world)
        cache_cpu = torch.zeros((1, case["pool_pages"], page, 1, 512), dtype=torch.bfloat16)
        rope_cache_cpu = torch.zeros((1, case["pool_pages"], page, 1, 64), dtype=torch.bfloat16)
        table = torch.zeros((padded, case["block_table_width"]), device="npu", dtype=torch.int32)
        for request in range(active):
            page_start = 1 + request * pages_per_request
            table[request, :pages_per_request] = torch.arange(
                page_start, page_start + pages_per_request, device="npu", dtype=torch.int32
            )
            first = page_start * page
            cache_cpu.view(-1, 512)[first : first + local_length].copy_(global_keys[request, owned_positions])
            rope_cache_cpu.view(-1, 64)[first : first + local_length].copy_(global_rope[request, owned_positions])
        cache_owner, rope_owner = cache_cpu.npu(), rope_cache_cpu.npu()
        cache, rope_cache = cache_owner[0], rope_owner[0]
        del cache_cpu, rope_cache_cpu
        if torch_npu.get_npu_format(cache) != 30 or torch_npu.get_npu_format(rope_cache) != 30:
            raise ValueError("DCP caches must retain the observed format30 layer view")
        query_lengths = (
            torch.arange(query_per_request, tokens + 1, query_per_request, device="npu", dtype=torch.int32)
            if case["phase"] == "prefill"
            else torch.arange(1, padded + 1, device="npu", dtype=torch.int32)
        )
        key_lengths = torch.tensor([local_length] * active + [0] * (padded - active), device="npu", dtype=torch.int32)
        indices_cpu = torch.full((tokens, 1, 2048), -1, dtype=torch.int32)
        for query_row in range(tokens if case["phase"] == "prefill" else active):
            visible = prefix + query_row % query_per_request + 1 if case["phase"] == "prefill" else length
            count = min(visible, 2048)
            indices_cpu[query_row, 0, :count] = torch.arange(visible - count, visible, dtype=torch.int32)
        indices = indices_cpu.npu()
        local_indices = remap_dcp_sparse_indices(indices, world, rank, page)
        expected_indices = torch.full_like(indices_cpu, -1)
        for query_row in range(tokens):
            candidates = indices_cpu[query_row, 0]
            selected = candidates[(candidates >= 0) & ((candidates // page) % world == rank)]
            owned = selected // (page * world) * page + selected % page
            expected_indices[query_row, 0, : owned.numel()] = owned
        if not torch.equal(local_indices.cpu(), expected_indices):
            raise ValueError("DCP execution numerical check failed: sparse index localization")
        metadata = SimpleNamespace(dcp_block_tables=table, dcp_seq_lens=key_lengths)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND if case["phase"] == "prefill" else ForwardMode.DECODE
        )

        def partial_attention():
            selected_indices = (
                remap_dcp_sparse_indices(indices, world, rank, page) if case["remap_topk"] else local_indices
            )
            gathered_query, gathered_rope = all_gather_q_for_mla_decode(query, query_rope)
            partial, lse = forward_dcp_sparse_attention(
                q_nope=gathered_query.contiguous(),
                q_rope=gathered_rope.contiguous(),
                k_nope=cache,
                k_rope=rope_cache,
                topk_indices=selected_indices,
                actual_seq_lengths_query=query_lengths,
                forward_metadata=metadata,
                forward_batch=forward_batch,
                speculative_num_draft_tokens=None,
                scaling=1 / 16,
            )
            partial = partial.reshape(tokens, heads * world, 512)
            return partial, lse

        def invoke():
            partial, lse = partial_attention()
            return cp_lse_ag_out_rs_mla_npu(partial, lse, group)

        def batch():
            return [invoke() for _ in range(protocol["iterations"])]

        distributed.barrier(group=group.cpu_group)
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
            raise ValueError("Unsupported DCP execution mode")

        def execute():
            if graph is None:
                return batch()
            if not args.negative_control:
                graph.replay()
            return outputs

        sign = 1
        selected_heads = sorted({0, heads - 1})
        columns = list(range(0, 512, 16))
        query_rows = sorted(
            {
                request * query_per_request + offset
                for request in {0, active - 1}
                for offset in {0, query_per_request - 1}
            }
        )
        head_indices = [source * heads + head for source in range(world) for head in selected_heads]
        query_samples = [None] * world
        distributed.all_gather_object(
            query_samples,
            {
                "query": query_cpu[query_rows][:, selected_heads].float(),
                "rope": rope_cpu[query_rows][:, selected_heads].float(),
            },
            group=group.cpu_group,
        )

        def verify():
            nonlocal sign
            sign = -sign
            query.neg_()
            query_rope.neg_()
            observed = execute()
            torch.npu.synchronize()
            partial, lse = partial_attention()
            torch.npu.synchronize()
            partial_cpu = partial[query_rows][:, head_indices][:, :, columns].float().cpu()
            lse_cpu = lse[query_rows][:, head_indices].float().cpu()
            checked, maximum, lse_error, ideal_error = 0, 0.0, 0.0, 0.0
            for position, query_row in enumerate(query_rows):
                request = query_row // query_per_request
                selected_indices = indices_cpu[query_row, 0]
                selected_indices = selected_indices[selected_indices >= 0].long()
                owned = selected_indices[(selected_indices // page) % world == rank]
                for source, sample in enumerate(query_samples):
                    begin_head = source * len(selected_heads)
                    actual_partial = partial_cpu[position, begin_head : begin_head + len(selected_heads)]
                    actual_lse = lse_cpu[position, begin_head : begin_head + len(selected_heads)]
                    if owned.numel() == 0:
                        if not ((actual_lse < 0) & (actual_lse.exp() == 0)).all():
                            raise ValueError("DCP execution numerical check failed: empty partial LSE")
                        continue
                    key = global_keys[request, owned].float()
                    rope = global_rope[request, owned].float()
                    scores = (sample["query"][position] @ key.T + sample["rope"][position] @ rope.T) * sign / 16
                    expected_partial = torch.softmax(scores, dim=-1) @ key[:, columns]
                    expected_lse = torch.logsumexp(scores, dim=-1)
                    difference = (actual_partial - expected_partial).abs()
                    if not torch.isfinite(actual_partial).all() or not torch.all(
                        difference <= 0.002 + 0.03 * expected_partial.abs()
                    ):
                        raise ValueError(
                            f"DCP execution numerical check failed: partial attention, max={difference.max().item()}"
                        )
                    current_lse_error = (actual_lse - expected_lse).abs()
                    if not torch.all(current_lse_error <= 0.005 + 0.001 * expected_lse.abs()):
                        raise ValueError("DCP execution numerical check failed: partial LSE")
                    maximum = max(maximum, difference.max().item())
                    lse_error = max(lse_error, current_lse_error.max().item())
                    checked += actual_partial.numel() + actual_lse.numel()
            partials = [None] * world
            distributed.all_gather_object(partials, {"partial": partial_cpu, "lse": lse_cpu}, group=group.cpu_group)
            first_head = rank * len(selected_heads)
            for position, query_row in enumerate(query_rows):
                rank_lse = torch.stack(
                    [value["lse"][position, first_head : first_head + len(selected_heads)] for value in partials]
                )
                rank_values = torch.stack(
                    [value["partial"][position, first_head : first_head + len(selected_heads)] for value in partials]
                )
                weights = torch.softmax(rank_lse, dim=0)
                rank_values = torch.where((weights == 0).unsqueeze(-1), 0, rank_values)
                expected = (rank_values * weights.unsqueeze(-1)).sum(0)
                request = query_row // query_per_request
                selected_indices = indices_cpu[query_row, 0]
                selected_indices = selected_indices[selected_indices >= 0].long()
                key = global_keys[request, selected_indices].float()
                rope = global_rope[request, selected_indices].float()
                queries = query_cpu[query_row, selected_heads].float() * sign
                ropes = rope_cpu[query_row, selected_heads].float() * sign
                scores = (queries @ key.T + ropes @ rope.T) / 16
                ideal = torch.softmax(scores, dim=-1) @ key[:, columns]
                for output in observed:
                    actual = output[query_row, selected_heads][:, columns].float().cpu()
                    difference = (actual - expected).abs()
                    if not torch.isfinite(actual).all() or not torch.all(difference <= 0.002 + 0.03 * expected.abs()):
                        raise ValueError(
                            f"DCP execution numerical check failed: merged attention, max={difference.max().item()}"
                        )
                    checked += actual.numel()
                    maximum = max(maximum, difference.max().item())
                    ideal_error = max(ideal_error, (actual - ideal).abs().max().item())
            return {
                "rank": rank,
                "sampled_elements": checked,
                "max_abs_error": maximum,
                "max_lse_error": lse_error,
                "ideal_fp32_max_abs_error": ideal_error,
                "input_mutation": sign,
                "reference": "Dense CPU partials and logsumexp, then CPU merge of checked BF16 partial outputs and FP32 LSE",
            }

        checks = [verify()]
        samples, wall_samples = [], []
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        for _ in range(protocol["repeats"]):
            distributed.barrier(group=group.cpu_group)
            started = time.perf_counter()
            begin.record()
            measured = execute()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / protocol["iterations"])
            wall_samples.append((time.perf_counter() - started) * 1000 / protocol["iterations"])
            del measured
        checks.append(verify())
        per_rank = [None] * world
        distributed.all_gather_object(
            per_rank,
            {"rank": rank, "samples_ms": samples, "wall_samples_ms": wall_samples, "checks": checks},
            group=group.cpu_group,
        )
        maximum_samples = [
            max(worker["samples_ms"][repeat] for worker in per_rank) for repeat in range(protocol["repeats"])
        ]
        variation = max(
            [
                statistics.pstdev(maximum_samples) / statistics.mean(maximum_samples),
                *[
                    statistics.pstdev(worker["samples_ms"]) / statistics.mean(worker["samples_ms"])
                    for worker in per_rank
                ],
            ]
        )
        row = {
            **case,
            "plan_index": index,
            "status": "passed",
            "rank_measurements": per_rank,
            "samples_ms": maximum_samples,
            "latency_ms": statistics.median(maximum_samples),
            "cv": variation,
            "execution_verification": "captured_graph_replay" if graph is not None else "eager_invocation",
            "cache_format": 30,
            "query_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "merge_dtype": "float32",
            "group_ranks": list(range(world)),
            "rank_layout": "contiguous",
            "ranks_per_node": world,
            "kernel_contract": "sglang_npu_dcp_sparse_chain/v1",
        }
        record["cases"].append(row)
        save()
        if rank == 0:
            print(json.dumps({"plan_index": index, "latency_ms": row["latency_ms"], "cv": variation}), flush=True)

    save()
    try:
        for index, case in selected:
            collect(index, case)
            gc.collect()
            torch.npu.empty_cache()
        record["status"] = "completed"
    except Exception:
        error = traceback.format_exc()
        args.output.with_suffix(f".rank-{rank}.failure.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "plan_index": index,
                    "case": case,
                    "error": error,
                    "collector_source_sha256": record["collector_source_sha256"],
                },
                indent=2,
            )
            + "\n"
        )
        record.update(status="failed", failed_plan_index=index, error=error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
