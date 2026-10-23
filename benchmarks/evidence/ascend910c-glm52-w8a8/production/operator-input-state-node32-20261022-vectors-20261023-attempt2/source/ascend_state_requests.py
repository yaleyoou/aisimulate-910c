"""Drive bounded state captures on the dedicated instrumented serving instance."""

import argparse
import json
import re
import urllib.request
from pathlib import Path


def request(base_url, endpoint, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    message = urllib.request.Request(base_url + endpoint, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(message, timeout=600) as response:
        data = response.read().decode()
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="glm-5-aisim")
    parser.add_argument("--case", default="8192_early")
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--skip-forwards", type=int, default=0)
    parser.add_argument("--max-forwards", type=int, default=3)
    args = parser.parse_args()
    if args.input_tokens < 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.case):
        raise ValueError("Use positive input tokens and a simple capture case name")
    output = args.capture_dir / args.case
    output.mkdir(parents=True, exist_ok=False)
    if args.skip_forwards + args.max_forwards > args.output_tokens:
        raise ValueError("Capture window extends beyond the request")
    if args.skip_forwards:
        installed = [json.loads(path.read_text()) for path in args.capture_dir.glob("installed-*.json")]
        if not installed or any(item.get("instrumentation_version", 1) < 2 for item in installed):
            raise ValueError("skip-forwards requires recording instrumentation version2")
    request(args.base_url, "/health")
    request(args.base_url, "/flush_cache", {})
    tokens = request(
        args.base_url,
        "/v1/tokenize",
        {
            "model": args.model,
            "prompt": "This is a bounded inference validation workload. ",
            "add_special_tokens": False,
        },
    )["tokens"]
    payload = {
        "input_ids": (tokens * ((args.input_tokens + len(tokens) - 1) // len(tokens)))[: args.input_tokens],
        "sampling_params": {"temperature": 0, "ignore_eos": True, "max_new_tokens": args.output_tokens},
        "routed_dp_rank": 0,
    }
    control = {
        "case_id": args.case,
        "enabled": True,
        "max_forwards": args.max_forwards,
        "skip_forwards": args.skip_forwards,
    }
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    control_path = args.capture_dir / "control.json"

    def write_control():
        temporary = control_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(control) + "\n")
        temporary.replace(control_path)

    write_control()
    try:
        response = request(args.base_url, "/generate", payload)
        metadata = response["meta_info"]
        if (metadata["prompt_tokens"], metadata["completion_tokens"], metadata["cached_tokens"]) != (
            args.input_tokens,
            args.output_tokens,
            0,
        ):
            raise ValueError("Observed workload does not match the capture request")
        (output / "request.json").write_text(json.dumps(payload) + "\n")
        (output / "response.json").write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"case_id": args.case, "meta_info": metadata}))
    finally:
        control["enabled"] = False
        write_control()


if __name__ == "__main__":
    main()
