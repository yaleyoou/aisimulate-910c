"""Separate safetensors header ordering from actual tensor payload identity."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="*.safetensors")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-tensor", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.model.glob(args.pattern))
    if not paths or args.output.exists():
        raise ValueError("Use existing tensor files and a fresh output")

    def inspect(path):
        before = path.stat()
        tensor_hashes = {}
        with path.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            if header_length > min(before.st_size - 8, 100_000_000):
                raise ValueError(f"Invalid header length: {path}")
            header = json.loads(stream.read(header_length))
            payload = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                payload.update(block)
            if args.per_tensor:
                for name, tensor in header.items():
                    if name == "__metadata__":
                        continue
                    start, end = tensor["data_offsets"]
                    stream.seek(8 + header_length + start)
                    remaining = end - start
                    digest = hashlib.sha256()
                    while remaining:
                        block = stream.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            raise ValueError(f"Truncated tensor {name}")
                        remaining -= len(block)
                        digest.update(block)
                    tensor_hashes[name] = {"dtype": tensor["dtype"], "shape": tensor["shape"], "sha256": digest.hexdigest()}
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f"File changed during read: {path}")
        tensors = {name: value for name, value in header.items() if name != "__metadata__"}
        return {
            "path": path.name, "bytes": before.st_size, "header_length": header_length,
            "tensor_descriptors_sha256": hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "payload_sha256": payload.hexdigest(), "tensors": len(tensors), "metadata": header.get("__metadata__"),
            "tensor_hashes": tensor_hashes,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(inspect, paths))
    result = {"schema": "aisimulate-tensor-payload-fingerprint/v1", "model": str(args.model), "files": files}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"files": len(files), "output": str(args.output)}))


if __name__ == "__main__":
    main()
