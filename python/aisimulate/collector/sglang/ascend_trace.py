"""Experimental observations from a live SGLang Ascend service, not FPM rows.

Uses the serving profiler API rather than substituting CUDA collector kernels.
The service must be dedicated to this campaign; cache flushing is explicit.
"""

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path


def request(base_url, endpoint, payload=None, timeout=1200):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base_url + endpoint, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        text = response.read().decode()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def validate_response(response, input_tokens, output_tokens):
    metadata = response["meta_info"]
    expected = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "cached_tokens": 0,
        "num_retractions": 0,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"{field}: expected {value}, observed {metadata.get(field)}")
    return metadata


def export_trace(container, remote, target):
    script = """
import pathlib, sys, tarfile
root = pathlib.Path(sys.argv[1])
files = sorted(root.rglob('*'))
done = [path for path in files if path.name == 'analyse.done']
if not done:
    raise RuntimeError('No completed profiler exports')
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for path in files:
        if path.is_file() and (
            path.suffix == '.csv'
            or path.name in ('trace_view.json', 'analyse.done', 'profiler_metadata.json')
            or path.name.startswith('profiler_info_')
        ):
            archive.add(path, arcname=str(path.relative_to(root)))
"""
    with target.open("wb") as stream:
        subprocess.run(
            ["docker", "exec", container, "python3", "-c", script, remote],
            stdout=stream,
            check=True,
        )
    with tarfile.open(target, "r:gz") as archive:
        members = archive.getnames()
    if not any(name.endswith("kernel_details.csv") for name in members):
        raise ValueError("Profiler archive contains no device kernel observations")


def collect(args):
    if not args.allow_flush_cache:
        raise ValueError("Collection needs --allow-flush-cache on a dedicated service")
    if args.output_tokens <= args.steps + 2:
        raise ValueError("Request must outlast bounded profiling and its flush step")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "aisimulate-ascend-serving-observations/v1",
        "status": "running",
        "timing_scope": "profiled serving kernels, not isolated operator benchmarks",
        "canonical_fpm_data": False,
        "cases": [],
        "limitations": [
            "Profiler perturbs timing; compare independent repetitions and controls.",
            "Only the captured forwards are covered, not the entire decode trajectory.",
            "Kernel duration sums are not forward latency when streams overlap.",
        ],
    }
    save(args.output / "manifest.json", manifest)
    inventory = {
        "container": args.container,
        "image_id": subprocess.check_output(
            ["docker", "inspect", "--format", "{{.Image}}", args.container], text=True
        ).strip(),
        "sglang_commit": subprocess.check_output(
            ["docker", "exec", args.container, "git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
    }
    save(args.output / "runtime-inventory.json", inventory)
    request(args.base_url, "/health", timeout=30)
    save(args.output / "server-info-before.json", request(args.base_url, "/get_server_info"))
    encoded = request(
        args.base_url,
        "/v1/tokenize",
        {
            "model": args.model,
            "prompt": "This is a bounded inference validation workload. ",
            "add_special_tokens": False,
        },
    )["tokens"]
    if not encoded or any(type(token) is not int for token in encoded):
        raise ValueError("Tokenizer did not return a nonempty flat token list")
    payload = {
        "input_ids": (encoded * ((args.input_tokens + len(encoded) - 1) // len(encoded)))[: args.input_tokens],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.output_tokens,
            "ignore_eos": True,
        },
        "routed_dp_rank": args.dp_rank,
    }
    save(args.output / "request.json", payload)
    profile_active = False
    try:
        for repeat in range(args.repeats + 1):
            case = args.output / ("control" if repeat == 0 else f"profile-{repeat}")
            case.mkdir()
            request(args.base_url, "/flush_cache", {})
            if repeat:
                remote = f"/tmp/aisim-ascend-{args.output.name}-{repeat}"
                profile = {
                    "output_dir": remote,
                    "num_steps": args.steps,
                    "activities": ["CPU", "GPU"],
                    "with_stack": False,
                    "record_shapes": True,
                    "detailed_annotations": True,
                    "profile_by_stage": False,
                    "profile_id": f"aisim-{args.output.name}-{repeat}",
                }
                save(case / "profile-request.json", profile)
                profile_active = True
                save(case / "start-profile.json", request(args.base_url, "/start_profile", profile))
            started = time.monotonic()
            response = request(args.base_url, "/generate", payload)
            elapsed = time.monotonic() - started
            save(case / "response.json", response)
            metadata = validate_response(response, args.input_tokens, args.output_tokens)
            if repeat:
                export_trace(args.container, remote, case / "trace.tar.gz")
                save(case / "stop-profile.json", {"mode": "automatic_num_steps", "completed_exports": True})
                profile_active = False
            manifest["cases"].append({"path": case.name, "elapsed_seconds": elapsed, "meta_info": metadata})
            save(args.output / "manifest.json", manifest)
            print(f"{case.name}: validated {args.input_tokens}/{args.output_tokens}", flush=True)
        manifest["status"] = "collected_pending_trace_validation"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if profile_active:
            try:
                save(args.output / "cleanup-stop-profile.json", request(args.base_url, "/stop_profile", {}))
            except Exception as error:
                manifest["cleanup_error"] = str(error)
        save(args.output / "manifest.json", manifest)
    request(args.base_url, "/health", timeout=30)
    save(args.output / "server-info-after.json", request(args.base_url, "/get_server_info"))
    hashes = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(path.relative_to(args.output))] = digest.hexdigest()
    save(args.output / "sha256.json", hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--container", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--allow-flush-cache", action="store_true")
    args = parser.parse_args()
    if min(args.input_tokens, args.output_tokens, args.repeats, args.steps) < 1:
        parser.error("token counts, repeats and steps must be positive")
    collect(args)


if __name__ == "__main__":
    main()
