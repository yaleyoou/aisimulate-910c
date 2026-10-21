import argparse
import datetime
import json
import pathlib
import time
import urllib.request


def request(base_url, route, payload=None, timeout=600):
    body = json.dumps(payload).encode() if payload is not None else None
    outgoing = urllib.request.Request(base_url + route, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(outgoing, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else {"http_status": response.status}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "base_url": args.base_url, "cases": [], "status": "in_progress"}
    result_path = args.output / "serving-smoke.json"

    def save():
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    save()
    try:
        result["health"] = request(args.base_url, "/health", timeout=15)
        server_info = request(args.base_url, "/get_server_info", timeout=30)
        (args.output / "server-info.json").write_text(json.dumps(server_info, indent=2) + "\n")
        for case_index in range(2):
            payload = {"model": "glm-5", "messages": [{"role": "user", "content": f"测试编号{case_index}：1加1等于几？请简短回答。"}], "max_tokens": 64, "temperature": 0}
            started = time.monotonic()
            response = request(args.base_url, "/v1/chat/completions", payload)
            if not response.get("choices"):
                raise ValueError("No chat completion choices returned")
            result["cases"].append({"case": f"short_chat_{case_index}", "elapsed_seconds": time.monotonic() - started, "response": response})
            save()
            print(f"short_chat_{case_index} returned", flush=True)
        encoded = request(args.base_url, "/v1/tokenize", {"model": "glm-5", "prompt": "This is a bounded inference validation workload. ", "add_special_tokens": False})
        unit_tokens = encoded["tokens"]
        if not unit_tokens or not all(isinstance(token, int) for token in unit_tokens):
            raise ValueError("Tokenizer did not return a non-empty flat token list")
        input_ids = (unit_tokens * ((8192 + len(unit_tokens) - 1) // len(unit_tokens)))[:8192]
        payload = {"input_ids": input_ids, "sampling_params": {"temperature": 0, "max_new_tokens": 1024, "ignore_eos": True}}
        (args.output / "request-8192-1024.json").write_text(json.dumps(payload) + "\n")
        started = time.monotonic()
        response = request(args.base_url, "/generate", payload)
        elapsed = time.monotonic() - started
        (args.output / "response-8192-1024.json").write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n")
        metadata = response["meta_info"]
        passed = metadata.get("prompt_tokens") == 8192 and metadata.get("completion_tokens") == 1024
        result["cases"].append({"case": "8192_1024", "elapsed_seconds": elapsed, "meta_info": metadata, "passed": passed})
        result["status"] = "passed" if passed else "failed"
        result["limits"] = "Single bounded functional request, not a statistically qualified performance benchmark or model accuracy evaluation. Short requests exercise scheduling but do not alone prove per-DP coverage."
        save()
        print("8192_1024", result["status"], flush=True)
        return 0 if passed else 1
    except Exception as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
