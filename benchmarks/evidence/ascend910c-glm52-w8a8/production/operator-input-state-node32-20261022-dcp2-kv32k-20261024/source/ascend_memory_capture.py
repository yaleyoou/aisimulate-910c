"""Observe initialized Ascend worker allocations without wrapping forward calls.

Audited SGLang baa9befa381511cd4d56f83980951e5a45fffc6d boundaries:
model_executor/model_runner.py:1251 load_model, :954 alloc_memory_pool,
and :1156 init_cuda_graphs. Snapshots surround these startup methods only.
Registered parameters/buffers and direct tensor attributes are inventoried
without reading their values. Allocation deltas are not additive estimates of
weights, communication and graph memory; consumers must reconcile lifetimes.
"""

import functools
import hashlib
import importlib
import json
import os
import subprocess
import time
from pathlib import Path


def install():
    directory = os.environ.get("AISIM_MEMORY_DIR")
    if not directory:
        raise ValueError("AISIM_MEMORY_DIR is required")
    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    if revision != "baa9befa381511cd4d56f83980951e5a45fffc6d":
        raise RuntimeError("Reaudit memory boundaries for this SGLang revision")

    import torch
    import torch_npu

    importlib.import_module("sglang.srt.layers.quantization")
    from sglang.srt.model_executor.model_runner import ModelRunner

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    def inventory(model):
        tensors = dict(model.named_parameters())
        tensors.update(model.named_buffers())
        for module_name, module in model.named_modules():
            for name, value in vars(module).items():
                if isinstance(value, torch.Tensor):
                    tensors[f"{module_name}.{name}".lstrip(".")] = value
        storages = {}
        records = []
        for name, tensor in sorted(tensors.items()):
            if tensor.device.type != "npu":
                records.append({"name": name, "device": str(tensor.device), "resident_npu": False})
                continue
            storage = tensor.untyped_storage()
            pointer = storage.data_ptr()
            storage_bytes = storage.nbytes()
            if pointer in storages and storages[pointer] != storage_bytes:
                raise ValueError("Conflicting storage sizes for the same NPU allocation")
            storages[pointer] = storage_bytes
            records.append(
                {
                    "name": name,
                    "device": str(tensor.device),
                    "resident_npu": True,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "logical_bytes": tensor.numel() * tensor.element_size(),
                    "storage_pointer": pointer,
                    "storage_bytes": storage_bytes,
                    "format": torch_npu.get_npu_format(tensor),
                }
            )
        return {"tensors": records, "unique_storages": len(storages), "unique_storage_bytes": sum(storages.values())}

    def snapshot(runner, boundary, include_inventory=False):
        if torch.npu.current_device() != runner.gpu_id:
            raise ValueError("Observer must use the worker's already selected device")
        torch.npu.synchronize()
        free_bytes, total_bytes = torch.npu.mem_get_info(runner.gpu_id)
        record = {
            "schema": "aisimulate-ascend-worker-memory/v1",
            "boundary": boundary,
            "rank": torch.distributed.get_rank(),
            "device": runner.gpu_id,
            "time_ns": time.time_ns(),
            "allocated_bytes": torch.npu.memory_allocated(runner.gpu_id),
            "reserved_bytes": torch.npu.memory_reserved(runner.gpu_id),
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "sglang_commit": revision,
            "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "runtime_field_units": {"pre_model_load_memory": "GiB", "graph_memory_usage": "GiB if scalar"},
            "inventory_scope": "PyTorch storage extents; not allocator reservations or complete non-model workspaces",
        }
        for name in ("pre_model_load_memory", "max_total_num_tokens", "max_running_requests", "graph_memory_usage"):
            value = getattr(runner, name, None)
            if value is None or isinstance(value, (bool, str, int, float)):
                record[name] = value
        if include_inventory:
            record["model_inventory"] = inventory(runner.model)
        pool = getattr(runner, "token_to_kv_pool", None)
        if pool is not None:
            record["kv_pool_class"] = type(pool).__module__ + "." + type(pool).__name__
            record["kv_pool_tensors"] = {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": value.numel() * value.element_size(),
                }
                for name, value in vars(pool).items()
                if isinstance(value, torch.Tensor)
            }
        path = root / f"rank-{record['rank']}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def wrap(name, include_inventory):
        original = getattr(ModelRunner, name)

        @functools.wraps(original)
        def observed(runner, *args, **kwargs):
            snapshot(runner, name + ":before")
            result = original(runner, *args, **kwargs)
            snapshot(runner, name + ":after", include_inventory)
            return result

        setattr(ModelRunner, name, observed)

    wrap("load_model", True)
    wrap("alloc_memory_pool", False)
    wrap("init_cuda_graphs", True)
