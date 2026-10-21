"""Fingerprint an unchanged checkpoint without loading tensor payloads into memory."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = sorted(path for path in args.model.rglob("*") if path.is_file())
    if not paths or any(path.is_symlink() for path in paths):
        raise ValueError("Expected a nonempty checkpoint without symlinked files")

    def fingerprint(path):
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f"Checkpoint file changed while hashing: {path.name}")
        return {"path": str(path.relative_to(args.model)), "bytes": after.st_size, "sha256": digest.hexdigest()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(fingerprint, paths))
    if paths != sorted(path for path in args.model.rglob("*") if path.is_file()):
        raise ValueError("Checkpoint file set changed while hashing")
    content = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "schema": "aisimulate-checkpoint-content/v1", "model_path": str(args.model),
        "checkpoint_sha256": hashlib.sha256(content).hexdigest(), "files": files,
        "total_bytes": sum(entry["bytes"] for entry in files),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"files": len(files), "total_bytes": result["total_bytes"], "checkpoint_sha256": result["checkpoint_sha256"]}))


if __name__ == "__main__":
    main()
