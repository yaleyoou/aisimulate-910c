"""Measure real Ascend KV replica transfers with source-selected rank mapping."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import socket
import statistics
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--store-url", required=True)
    parser.add_argument("--source-tp", type=int, required=True)
    parser.add_argument("--destination-tp", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior observations")
    plan = json.loads(args.plan.read_text())
    protocol = plan["protocol"]
    if any(type(value) is not int or value < 1 for value in protocol.values()) or protocol["repeats"] < 3:
        raise ValueError("PD timing requires positive protocol counts and at least three samples")
    setup = json.loads(args.setup.read_text())
    if plan["schema"] != "aisimulate-ascend-pd-plan/v1":
        raise ValueError("Unsupported PD plan")
    for name, value in setup["environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    os.environ["ASCEND_MF_STORE_URL"] = args.store_url
    os.environ.pop("ASCEND_MF_TRANSFER_PROTOCOL", None)
    local_rank = int(os.environ["LOCAL_RANK"])
    os.sched_setaffinity(0, setup["worker_cpu_affinity"][str(local_rank)])

    import numpy as np
    import torch
    import torch.distributed as distributed
    import torch_npu
    from sglang.srt.disaggregation.ascend.conn import AscendKVManager
    from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
    from sglang.srt.disaggregation.base.conn import KVArgs
    from sglang.srt.disaggregation.common.conn import CommonKVManager
    from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
    from sglang.srt.disaggregation.mooncake.utils import check_mooncake_custom_mem_pool_enabled
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    init_npu_backend()
    torch.npu.set_device(local_rank)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    if revision != "baa9befa381511cd4d56f83980951e5a45fffc6d":
        raise ValueError("Reaudit PD source for another runtime revision")
    distributed.init_process_group("gloo", timeout=datetime.timedelta(seconds=600))
    rank, world = distributed.get_rank(), distributed.get_world_size()
    if world != 32 or int(os.environ["LOCAL_WORLD_SIZE"]) != 16:
        raise ValueError("PD calibration requires two16-device nodes")
    prefill = rank < 16
    local_tp = args.source_tp if prefill else args.destination_tp
    selected = [
        (index, case)
        for index, case in enumerate(plan["cases"])
        if case["source_tp"] == args.source_tp
        and case["destination_tp"] == args.destination_tp
        and (args.case_index is None or args.case_index == index)
    ]
    if not selected or (args.negative_control and len(selected) != 1):
        raise ValueError("Empty selection or invalid negative control")
    publish(
        ServerArgs(
            model_path=str(args.model_metadata),
            device="npu",
            dtype="bfloat16",
            quantization="modelslim",
            tp_size=16,
            ep_size=16,
            dp_size=16 // local_tp,
            enable_dp_attention=True,
            enable_dp_lm_head=True,
            disable_shared_experts_fusion=True,
            context_length=180000,
            moe_a2a_backend="deepep",
            disaggregation_mode="prefill" if prefill else "decode",
            disaggregation_transfer_backend="ascend",
            attention_backend="ascend",
        ),
        role="scheduler",
    )
    record = {
        "schema": "aisimulate-ascend-pd-observations/v1",
        "status": "initializing",
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "framework_commit": revision,
        "scope": plan["scope"],
        "protocol": plan["protocol"],
        "serving_setup": setup,
        "selected_plan_indices": [index for index, _ in selected],
        "negative_control": args.negative_control,
        "cases": [],
    }

    def save():
        if rank == 0:
            args.output.write_text(json.dumps(record, indent=2) + "\n")

    save()
    engine = None
    try:
        pool = NPUMLATokenToKVPool(
            size=53248,
            page_size=128,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=78,
            device="npu",
            enable_memory_saver=False,
            index_head_dim=128,
        )
        buffers = (pool.k_buffer, pool.v_buffer, pool.index_k_buffer)
        pointers, lengths, item_lengths = pool.get_contiguous_buf_infos()
        if len(pointers) != 234 or sum(item_lengths) != 14_057_472:
            raise ValueError("Actual memory-pool layout differs from the plan")
        engine = AscendTransferEngine(
            args.local_ip, local_rank, DisaggregationMode.PREFILL if prefill else DisaggregationMode.DECODE
        )
        manager = AscendKVManager.__new__(AscendKVManager)
        manager.engine, manager.pp_size, manager.dcp_size = engine, 1, 1
        manager.is_mla_backend, manager.is_hybrid_mla_backend = True, False
        manager.enable_custom_mem_pool, manager.custom_mem_pool_type = check_mooncake_custom_mem_pool_enabled()
        if manager.enable_custom_mem_pool:
            raise ValueError("Custom allocator is outside this producer contract")
        kv_args = KVArgs()
        kv_args.kv_data_ptrs, kv_args.kv_data_lens, kv_args.kv_item_lens = pointers, lengths, item_lengths
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.state_data_ptrs, kv_args.state_data_lens = [], [], [], []
        kv_args.kv_buf_groups = 3
        manager.kv_args = kv_args
        manager.register_buffer_to_engine()
        peers = [None] * world
        identity = {
            "rank": rank,
            "local_rank": local_rank,
            "host": socket.gethostname(),
            "session": engine.session_id,
            "pointers": pointers,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "memfabric": importlib.metadata.version("memfabric-hybrid"),
            "hardware": torch.npu.get_device_name(local_rank),
            "driver": Path("/usr/local/Ascend/driver/version.info").read_text(),
            "source_sha256": record["collector_source_sha256"],
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "plan_sha256": record["plan_sha256"],
            "selected_plan_indices": record["selected_plan_indices"],
            "source_tp": args.source_tp,
            "destination_tp": args.destination_tp,
            "pool_shapes": [list(buffer.shape) for buffer in buffers],
            "pool_formats": [torch_npu.get_npu_format(buffer) for buffer in buffers],
        }
        distributed.all_gather_object(peers, identity)
        if any(
            peer["source_sha256"] != record["collector_source_sha256"]
            or peer["plan_sha256"] != record["plan_sha256"]
            or peer["selected_plan_indices"] != record["selected_plan_indices"]
            for peer in peers
        ):
            raise ValueError("Participants use different source or selected plan")
        record["devices"] = [{name: value for name, value in peer.items() if name != "pointers"} for peer in peers]
        mappings = []
        for destination in range(args.destination_tp):
            context = SimpleNamespace(
                attn_tp_size=args.destination_tp,
                is_mla_backend=True,
                is_hybrid_mla_backend=False,
                kv_args=SimpleNamespace(engine_rank=destination),
                attn_cp_size=1,
                attn_cp_rank=0,
                pp_size=1,
                pp_rank=0,
            )
            information = SimpleNamespace(attn_tp_size=args.source_tp, attn_cp_size=1, pp_size=1)
            CommonKVManager._resolve_rank_mapping(context, information)
            mappings.append({"source_rank": information.target_tp_rank, "destination_rank": 16 + destination})
        record["rank_mapping"] = mappings
        record["status"] = "collecting"
        save()
        destinations = [row["destination_rank"] for row in mappings if row["source_rank"] == rank]
        receiver = any(row["destination_rank"] == rank for row in mappings)

        def fill(nonce):
            for component, buffer in enumerate(buffers):
                if not prefill:
                    buffer.zero_()
                    continue
                layers = torch.arange(78, device="npu").view(78, 1, 1, 1, 1)
                pages = torch.arange(buffer.shape[1], device="npu").view(1, -1, 1, 1, 1)
                buffer.copy_(((layers * 7 + pages * 13 + component * 11 + nonce) % 127).to(torch.bfloat16))
                buffer[..., 0].copy_((pages % 127).squeeze(-1).to(torch.bfloat16))
                buffer[..., 1].copy_((pages // 127).squeeze(-1).to(torch.bfloat16))
                buffer[..., 2].copy_(layers.squeeze(-1).to(torch.bfloat16))
                buffer[..., 3].fill_(component + nonce)
                buffer[..., 4].copy_(torch.arange(128, device="npu").to(torch.bfloat16).view(1, 1, 128, 1))
            torch.npu.synchronize()
            distributed.barrier()

        def digest(indices):
            result = hashlib.sha256()
            for buffer in buffers:
                for layer in range(78):
                    result.update(
                        buffer[layer].index_select(0, indices).cpu().contiguous().view(torch.int16).numpy().tobytes()
                    )
            return result.hexdigest()

        with ThreadPoolExecutor(max_workers=1) as executor:
            for index, case in selected:
                if (
                    case["component_widths"] != [512, 64, 128]
                    or case["num_layers"] != 78
                    or case["page_size"] != 128
                    or case["reference_pool_tokens"] != 53248
                    or case["dtype"] != "bfloat16"
                    or case["protocol"] != "sdma"
                    or case["active_requests"] != 1
                ):
                    raise ValueError("Unsupported PD geometry")
                pages = case["tokens"] // 128
                if case["page_layout"] == "contiguous":
                    source_pages = np.arange(1, pages + 1, dtype=np.int32)
                    destination_pages = np.arange(pages + 1, 2 * pages + 1, dtype=np.int32)
                elif case["page_layout"] == "fragmented":
                    source_pages = np.random.default_rng(31).permutation(np.arange(1, 417, dtype=np.int32))[:pages]
                    destination_pages = np.random.default_rng(32).permutation(np.arange(1, 417, dtype=np.int32))[:pages]
                else:
                    raise ValueError("Unknown page layout")
                source_indices = torch.tensor(source_pages, device="npu", dtype=torch.int64)
                destination_indices = torch.tensor(destination_pages, device="npu", dtype=torch.int64)
                source_runs, destination_runs = group_concurrent_contiguous(source_pages, destination_pages)
                if len(source_runs) != len(destination_runs):
                    raise ValueError("Source and destination transfer runs disagree")
                payload_bytes = sum(item_lengths) * pages

                def send():
                    if args.negative_control:
                        return
                    for destination in destinations:
                        result = manager.send_kvcache(
                            peers[destination]["session"],
                            source_pages,
                            peers[destination]["pointers"],
                            destination_pages,
                            executor,
                        )
                        if result != 0:
                            raise RuntimeError(f"Actual Ascend KV transfer returned {result}")

                def verify(nonce):
                    fill(nonce)
                    expected = [digest(source_indices) if rank == 0 else None]
                    distributed.broadcast_object_list(expected, src=0)
                    send()
                    distributed.barrier()
                    torch.npu.synchronize()
                    actual = digest(destination_indices) if receiver else expected[0]
                    valid = actual == expected[0] and not (
                        receiver and any(torch.count_nonzero(buffer[:, 0]).item() for buffer in buffers)
                    )
                    valid_ranks = [None] * world
                    distributed.all_gather_object(valid_ranks, valid)
                    if not all(valid_ranks):
                        raise ValueError("PD full-payload or sentinel numerical check failed")
                    distributed.barrier()
                    return {"sha256": actual, "checked_bytes": payload_bytes if receiver else 0}

                checks = [verify(index * 2 + 1)]
                for _ in range(plan["protocol"]["warmup"]):
                    distributed.barrier()
                    send()
                samples, call_samples = [], []
                for _ in range(plan["protocol"]["repeats"]):
                    calls = []
                    for _ in range(plan["protocol"].get("iterations", 1)):
                        distributed.barrier()
                        started = time.perf_counter()
                        send()
                        calls.append((time.perf_counter() - started) * 1000 if destinations else 0.0)
                    call_samples.append(calls)
                    samples.append(statistics.mean(calls))
                checks.append(verify(index * 2 + 2))
                rank_results = [None] * world
                distributed.all_gather_object(
                    rank_results,
                    {
                        "rank": rank,
                        "destinations": destinations,
                        "samples_ms": samples,
                        "call_samples_ms": call_samples,
                        "checks": checks,
                    },
                )
                critical = [
                    max(value["samples_ms"][sample] for value in rank_results)
                    for sample in range(plan["protocol"]["repeats"])
                ]
                record["cases"].append(
                    {
                        **case,
                        "plan_index": index,
                        "latency_ms": statistics.median(critical),
                        "cv": statistics.pstdev(critical) / statistics.mean(critical),
                        "rank_results": rank_results,
                        "payload_bytes_per_replica": payload_bytes,
                        "total_wire_bytes": payload_bytes * args.destination_tp,
                        "segments_per_buffer": len(source_runs),
                        "source_pages": source_pages.tolist(),
                        "destination_pages": destination_pages.tolist(),
                        "kernel_source": "sglang.srt.disaggregation.ascend.conn.AscendKVManager.send_kvcache",
                    }
                )
                save()
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "plan_index": index,
                                "tokens": case["tokens"],
                                "layout": case["page_layout"],
                                "latency_ms": record["cases"][-1]["latency_ms"],
                            }
                        ),
                        flush=True,
                    )
        record["status"] = "completed"
        save()
    except Exception:
        record.update(status="failed", error=traceback.format_exc())
        save()
        raise
    finally:
        if engine is not None:
            engine.engine.destroy()
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
