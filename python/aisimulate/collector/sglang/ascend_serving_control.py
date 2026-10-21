"""Measure dedicated serving requests with frozen token identities and no profiler.

Times are client-observed HTTP stream arrivals, including scheduling, transport
and frontend work. They are not isolated forward or operator latencies.
"""

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
import traceback
import urllib.request
from pathlib import Path

from ascend_trace import request, validate_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--allow-flush-cache", action="store_true")
    args = parser.parse_args()
    if not args.allow_flush_cache or args.repeats < 1 or args.warmup < 0:
        parser.error("Use a dedicated service, explicit cache flushing and positive repeats")
    payloads = json.loads(args.requests.read_text())
    if not isinstance(payloads, list) or not payloads or len(payloads) > 10:
        raise ValueError("This campaign supports1..10frozen requests within the measured KV budget")
    for payload in payloads:
        if (
            not isinstance(payload.get("input_ids"), list)
            or len(payload["input_ids"]) != 8192
            or any(type(token) is not int or token < 0 for token in payload["input_ids"])
            or payload["sampling_params"].get("max_new_tokens") != 1024
            or payload["sampling_params"].get("temperature") != 0
            or payload["sampling_params"].get("ignore_eos") is not True
            or payload.get("routed_dp_rank") not in (0, 1)
        ):
            raise ValueError("Frozen requests must match the8192/1024deterministic campaign")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "requests.json").write_bytes(args.requests.read_bytes())
    manifest = {
        "schema": "aisimulate-ascend-serving-control/v1",
        "status": "running",
        "scope": "Client-observed HTTP stream timings; no operator/forward timing claim",
        "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "concurrency": len(payloads),
        "warmup_trials": args.warmup,
        "measurement_trials": args.repeats,
        "trials": [],
    }

    def save():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def generate(index, barrier, trial):
        payload = {**payloads[index], "stream": True}
        message = urllib.request.Request(
            args.base_url + "/generate", json.dumps(payload).encode(), {"Content-Type": "application/json"}
        )
        barrier.wait(timeout=30)
        start = time.monotonic_ns()
        arrivals = []
        final = None
        completed = 0
        with urllib.request.urlopen(message, timeout=600) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                final = json.loads(line[5:].strip())
                tokens = final["meta_info"]["completion_tokens"]
                if tokens < completed or tokens > 1024:
                    raise ValueError("Streaming completion count regressed or exceeded the request")
                if tokens > completed:
                    arrivals.append([tokens, (time.monotonic_ns() - start) / 1e6])
                    completed = tokens
        if final is None or len(arrivals) < 2:
            raise ValueError("Insufficient streaming observations")
        validate_response(final, 8192, 1024)
        record = {
            "request_index": index,
            "start_monotonic_ns": start,
            "end_monotonic_ns": time.monotonic_ns(),
            "arrivals_tokens_ms": arrivals,
            "ttft_ms": arrivals[0][1],
            "first_chunk_tokens": arrivals[0][0],
            "tpot_ms": (arrivals[-1][1] - arrivals[0][1]) / (arrivals[-1][0] - arrivals[0][0]),
            "response": final,
        }
        (trial / f"request-{index}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        return {key: value for key, value in record.items() if key not in {"response", "arrivals_tokens_ms"}}

    save()
    try:
        request(args.base_url, "/health", timeout=30)
        for ordinal in range(args.warmup + args.repeats):
            warmup = ordinal < args.warmup
            trial = args.output / f"trial-{ordinal}"
            trial.mkdir()
            flushed = request(args.base_url, "/flush_cache", {})
            (trial / "flush.json").write_text(json.dumps(flushed) + "\n")
            barrier = threading.Barrier(len(payloads))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(payloads)) as pool:
                records = list(pool.map(lambda index: generate(index, barrier, trial), range(len(payloads))))
            duration = (
                max(record["end_monotonic_ns"] for record in records)
                - min(record["start_monotonic_ns"] for record in records)
            ) / 1e9
            manifest["trials"].append(
                {
                    "ordinal": ordinal,
                    "warmup": warmup,
                    "requests": records,
                    "duration_seconds": duration,
                    "output_tokens_per_second": 1024 * len(records) / duration,
                }
            )
            save()
            print(json.dumps({"trial": ordinal, "warmup": warmup, "duration_seconds": duration}), flush=True)
        manifest["status"] = "completed"
    except Exception:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
