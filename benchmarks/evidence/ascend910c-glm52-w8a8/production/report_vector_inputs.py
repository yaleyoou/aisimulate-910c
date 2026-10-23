"""Validate source-state captures and extract compact invocation signatures."""

import collections
import hashlib
import json
from pathlib import Path


def compact(value):
    if isinstance(value, dict):
        return {name: compact(item) for name, item in value.items() if name not in {"data_ptr", "device", "parameter", "module_id", "module", "captured_values"}}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def main():
    root = Path(__file__).resolve().parent / "operator-input-state-node32-20261022-vectors-20261023-attempt2"
    campaign = json.loads((root / "request-campaign.json").read_text())
    if campaign["status"] != "completed" or len(campaign["cases"]) != 5:
        raise ValueError("Incomplete vector-state campaign")
    expected = {"vectors8192_c1": (1, 3), "vectors8192_c4": (4, 6), "vectors128_c18": (18, 3), "vectors128_c34": (34, 3), "vectors128_c50": (50, 3)}
    report = {"status": "validated_state_observations", "scope": "Actual invocation metadata; not latency data or full-model accuracy", "requests": 0, "forward_records": 0, "cases": {}, "sources": {}, "limits": ["The first C4 record on each rank belongs to the driver's health check and is excluded from workload signatures.", "The pinned ModelSlim RMSNorm bias paths require source auditing; their main boundary calls were not included in this hook set.", "Startup graph memory includes the state observer's allocations."]}
    signatures = collections.defaultdict(dict)
    for name, (requests, forwards) in expected.items():
        directory = root / "capture" / name
        paths = {int(path.stem.split("-")[-1]): path for path in directory.glob("inputs-rank-*.jsonl")}
        if set(paths) != set(range(16)):
            raise ValueError(f"Missing input ranks: {name}")
        if requests == 1:
            responses = [json.loads((directory / "response.json").read_text())["meta_info"]]
        else:
            workload = json.loads((directory / "workload-validation.json").read_text())
            if workload["status"] != "passed":
                raise ValueError("Request validation failed")
            responses = workload["requests"]
        if len(responses) != requests or any(record["cached_tokens"] or record.get("num_retractions", 0) for record in responses):
            raise ValueError("Unexpected request reuse/retraction")
        summary = {"requests": requests, "forwards_per_rank": forwards, "rank0": []}
        for rank, path in sorted(paths.items()):
            raw = path.read_bytes()
            records = [json.loads(line) for line in raw.splitlines()]
            if len(records) != forwards or [record["ordinal"] for record in records] != list(range(forwards)):
                raise ValueError(f"Incomplete forward sequence: {name}/{rank}")
            report["sources"][str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
            report["forward_records"] += len(records)
            for record in records:
                if record["schema"] != "aisimulate-ascend-operator-inputs/v1" or record["rank"] != rank:
                    raise ValueError("Input record identity mismatch")
                if name == "vectors8192_c4" and record["ordinal"] == 0:
                    continue
                if rank == 0:
                    summary["rank0"].append({key: record[key] for key in ("ordinal", "execution", "batch_size", "global_num_tokens_cpu")})
                    for call in record["calls"]:
                        if call["kernel"].startswith(("sglang.srt.hardware_backend.npu.quantization", "DeepEPDispatcher", "sglang.srt.hardware_backend.npu.moe.topk")):
                            continue
                        signature = compact({"arguments": call["arguments"], "output": call["output"]})
                        identity = json.dumps(signature, sort_keys=True, separators=(",", ":"))
                        signatures[call["kernel"]].setdefault(identity, {"execution": record["execution"], "case": name, "signature": signature})
        report["requests"] += requests
        report["cases"][name] = summary
    report["invocations"] = {kernel: list(values.values()) for kernel, values in sorted(signatures.items())}
    (root / "vector-state-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "requests", "forward_records")}))


if __name__ == "__main__":
    main()
