"""Capture steady decode routing after all dedicated requests are admitted."""

import argparse
import concurrent.futures
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

from ascend_state_requests import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="glm-5-aisim")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-tokens", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 10 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.case):
        raise ValueError("Use a fresh simple case name and concurrency1..10 for this verified KV budget")
    output = args.capture_dir / args.case
    output.mkdir(parents=True, exist_ok=False)
    control = {"case_id": args.case, "enabled": False, "max_forwards": 3}

    def save_control():
        target = args.capture_dir / "control.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(control) + "\n")
        temporary.replace(target)

    save_control()
    request(args.base_url, "/health")
    request(args.base_url, "/flush_cache", {})
    ready = [threading.Event() for _ in range(args.concurrency)]
    payloads = []
    for index in range(args.concurrency):
        tokens = request(
            args.base_url,
            "/v1/tokenize",
            {
                "model": args.model,
                "prompt": f"Document {index}: Explain a practical example about topic {index * 17 + 3}. ",
                "add_special_tokens": False,
            },
        )["tokens"]
        payloads.append(
            {
                "input_ids": (tokens * ((8192 + len(tokens) - 1) // len(tokens)))[:8192],
                "sampling_params": {"temperature": 0, "ignore_eos": True, "max_new_tokens": args.output_tokens},
                "routed_dp_rank": index % 2,
                "stream": True,
            }
        )
    (output / "requests.json").write_text(json.dumps(payloads) + "\n")

    def generate(index):
        message = urllib.request.Request(
            args.base_url + "/generate", json.dumps(payloads[index]).encode(), {"Content-Type": "application/json"}
        )
        final = None
        with urllib.request.urlopen(message, timeout=600) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                final = json.loads(payload)
                if "meta_info" not in final:
                    raise ValueError(f"Unexpected streaming response: {final}")
                if final["meta_info"].get("completion_tokens", 0) >= 1:
                    ready[index].set()
        if final is None:
            raise ValueError("Empty streaming response")
        (output / f"response-{index}.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
        metadata = final["meta_info"]
        if (
            metadata["prompt_tokens"],
            metadata["completion_tokens"],
            metadata["cached_tokens"],
            metadata["num_retractions"],
        ) != (8192, args.output_tokens, 0, 0):
            raise ValueError("The observed request did not match the state-capture workload")
        return metadata

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(generate, index) for index in range(args.concurrency)]
            deadline = time.monotonic() + 300
            while not all(event.is_set() for event in ready):
                for future in futures:
                    if future.done():
                        future.result()
                if time.monotonic() > deadline:
                    raise TimeoutError("All requests were not admitted before the capture deadline")
                time.sleep(0.05)
            if any(future.done() for future in futures):
                raise ValueError("A request finished before the steady decode capture could start")
            control["enabled"] = True
            save_control()
            metadata = [future.result() for future in futures]
            (output / "workload-validation.json").write_text(
                json.dumps({"status": "passed", "requests": metadata}, indent=2) + "\n"
            )
            print(json.dumps({"case": args.case, "requests": len(metadata), "status": "passed"}))
    finally:
        control["enabled"] = False
        save_control()


if __name__ == "__main__":
    main()
