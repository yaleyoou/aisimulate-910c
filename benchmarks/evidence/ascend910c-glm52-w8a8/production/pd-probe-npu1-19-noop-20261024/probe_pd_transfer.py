"""Verify the installed Ascend KV send path between one prefill/decode rank pair."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--store-url", required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not 0 < args.tokens <= 32768 or args.tokens % 128:
        parser.error("Use a fresh output and a positive page-aligned token count")
    setup = json.loads(args.setup.read_text())
    for name, value in setup["environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    os.environ["ASCEND_MF_STORE_URL"] = args.store_url
    os.environ.pop("ASCEND_MF_TRANSFER_PROTOCOL", None)
    os.sched_setaffinity(0, setup["worker_cpu_affinity"]["0"])

    import numpy as np
    import torch
    import torch.distributed as distributed
    import torch_npu

    from sglang.srt.disaggregation.ascend.conn import AscendKVManager
    from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
    from sglang.srt.disaggregation.base.conn import KVArgs
    from sglang.srt.disaggregation.mooncake.utils import check_mooncake_custom_mem_pool_enabled
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    init_npu_backend()
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    if revision != "baa9befa381511cd4d56f83980951e5a45fffc6d":
        raise ValueError("Reaudit the PD implementation for another framework revision")
    distributed.init_process_group("gloo", timeout=datetime.timedelta(seconds=300))
    rank = distributed.get_rank()
    if distributed.get_world_size() != 2:
        raise ValueError("This probe requires one process on each of two nodes")
    role = DisaggregationMode.PREFILL if rank == 0 else DisaggregationMode.DECODE
    publish(ServerArgs(
        model_path=str(args.model_metadata), device="npu", dtype="bfloat16", quantization="modelslim",
        tp_size=16, ep_size=16, dp_size=2, enable_dp_attention=True, enable_dp_lm_head=True,
        disable_shared_experts_fusion=True, context_length=180000, moe_a2a_backend="deepep",
        disaggregation_mode="prefill" if rank == 0 else "decode", disaggregation_transfer_backend="ascend",
        attention_backend="ascend",
    ), role="scheduler")
    record = {
        "schema": "aisimulate-ascend-pd-probe/v1", "status": "initializing", "rank": rank,
        "scope": "One rank pair; actual memory-pool and send_kvcache path; excludes bootstrap/admission and node-wide contention",
        "framework_commit": revision, "memfabric_version": importlib.metadata.version("memfabric-hybrid"),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "driver": Path("/usr/local/Ascend/driver/version.info").read_text(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokens": args.tokens, "protocol": "sdma", "negative_control": args.negative_control,
    }

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    save()
    engine = None
    try:
        pool = NPUMLATokenToKVPool(
            size=53248, page_size=128, dtype=torch.bfloat16, kv_lora_rank=512, qk_rope_head_dim=64,
            layer_num=78, device="npu", enable_memory_saver=False, index_head_dim=128,
        )
        buffers = (pool.k_buffer, pool.v_buffer, pool.index_k_buffer)
        pointers, lengths, item_lengths = pool.get_contiguous_buf_infos()
        if len(pointers) != 234 or sum(item_lengths) != 14_057_472:
            raise ValueError("The actual DSA pool differs from its measured BF16 layout")
        engine = AscendTransferEngine(args.local_ip, 0, role)
        manager = AscendKVManager.__new__(AscendKVManager)
        manager.engine = engine
        manager.pp_size = manager.dcp_size = 1
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool, manager.custom_mem_pool_type = check_mooncake_custom_mem_pool_enabled()
        if manager.enable_custom_mem_pool:
            raise ValueError("The probe has not audited a custom allocator")
        kv_args = KVArgs()
        kv_args.kv_data_ptrs, kv_args.kv_data_lens, kv_args.kv_item_lens = pointers, lengths, item_lengths
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.state_data_ptrs, kv_args.state_data_lens = [], [], [], []
        kv_args.kv_buf_groups = 3
        manager.kv_args = kv_args
        manager.register_buffer_to_engine()
        peers = [None, None]
        distributed.all_gather_object(peers, {
            "session": engine.session_id, "pointers": pointers, "source_sha256": record["source_sha256"],
            "tokens": args.tokens,
        })
        if any(peer["source_sha256"] != record["source_sha256"] or peer["tokens"] != args.tokens for peer in peers):
            raise ValueError("Transfer peers have different source or workload")
        count = args.tokens // 128
        source_pages = np.arange(1, count + 1, dtype=np.int32)
        destination_pages = np.arange(count + 1, 2 * count + 1, dtype=np.int32) if 2 * count < 417 else source_pages
        source_indices = torch.tensor(source_pages, device="npu", dtype=torch.int64)
        destination_indices = torch.tensor(destination_pages, device="npu", dtype=torch.int64)
        record.update(status="registered", pool_shapes=[list(buffer.shape) for buffer in buffers],
                      pool_formats=[torch_npu.get_npu_format(buffer) for buffer in buffers],
                      transfer_bytes=int(sum(item_lengths) * count), registration_bytes=sum(lengths))
        save()

        def prepare(nonce):
            for component, buffer in enumerate(buffers):
                if rank != 0:
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
            value = hashlib.sha256()
            for buffer in buffers:
                for layer in range(78):
                    view = buffer[layer].index_select(0, indices).cpu().contiguous().view(torch.int16)
                    value.update(view.numpy().tobytes())
            return value.hexdigest()

        with ThreadPoolExecutor(max_workers=1) as executor:
            def send():
                if rank == 0:
                    if args.negative_control:
                        return
                    result = manager.send_kvcache(
                        peers[1]["session"], source_pages, peers[1]["pointers"], destination_pages, executor
                    )
                    if result != 0:
                        raise RuntimeError(f"Actual Ascend KV transfer returned {result}")

            def verify(nonce):
                prepare(nonce)
                expected = [digest(source_indices) if rank == 0 else None]
                distributed.broadcast_object_list(expected, src=0)
                send()
                distributed.barrier()
                torch.npu.synchronize()
                actual = digest(destination_indices) if rank == 1 else expected[0]
                if actual != expected[0] or (rank == 1 and any(torch.count_nonzero(buffer[:, 0]).item() for buffer in buffers)):
                    raise ValueError("PD full-payload or sentinel numerical check failed")
                return {"nonce": nonce, "sha256": actual, "checked_bytes": record["transfer_bytes"]}
            checks = [verify(1)]
            for _ in range(3):
                distributed.barrier()
                send()
            samples = []
            for _ in range(7):
                distributed.barrier()
                started = time.perf_counter()
                send()
                if rank == 0:
                    samples.append((time.perf_counter() - started) * 1000)
            checks.append(verify(2))
            distributed.barrier()
        record.update(status="verified", checks=checks, samples_ms=samples)
        if samples:
            record.update(latency_ms=statistics.median(samples), cv=statistics.pstdev(samples) / statistics.mean(samples))
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
