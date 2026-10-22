"""Prepare a private shared checkpoint and verify every destination file."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.umask(0o077)
    manifest = json.loads(args.manifest.read_text())
    args.destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.source.resolve() == args.destination.resolve():
        raise ValueError("Destination must be a separate private copy")
    if shutil.disk_usage(args.destination).free < manifest["total_bytes"]:
        raise ValueError("Insufficient space for a complete verified checkpoint")
    state = {"status": "running", "source": str(args.source), "destination": str(args.destination),
             "expected_sha256": manifest["checkpoint_sha256"], "started_at_unix": time.time(), "files": []}

    def save():
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.report)

    def copy(entry):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest path escapes checkpoint")
        source, target = args.source / relative, args.destination / relative
        if source.is_symlink() or source.stat().st_size != entry["bytes"]:
            raise ValueError(f"Source changed: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        existing = target.exists()
        candidate = target if existing else target.with_name(target.name + ".partial")
        if not existing:
            with source.open("rb") as incoming, candidate.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if candidate.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Destination content mismatch: {relative}")
        if not existing:
            candidate.replace(target)
        return {**entry, "verified_existing": existing}

    save()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = [pool.submit(copy, entry) for entry in manifest["files"]]
        for future in concurrent.futures.as_completed(futures):
            state["files"].append(future.result())
            save()
            print(json.dumps({"verified_files": len(state["files"]), "total": len(futures)}), flush=True)
        files = [{key: row[key] for key in ("path", "bytes", "sha256")} for row in sorted(state["files"], key=lambda row: row["path"])]
        digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != manifest["checkpoint_sha256"]:
            raise ValueError("Checkpoint manifest identity mismatch")
        state.update(status="completed", checkpoint_sha256=digest, elapsed_seconds=time.time() - state["started_at_unix"])
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        save()


if __name__ == "__main__":
    main()
