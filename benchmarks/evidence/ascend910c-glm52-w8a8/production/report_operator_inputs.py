"""Validate rank coverage and summarize actual GLM operator input contracts."""

import collections
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent / "operator-input-state-node31-20261022-attempt2"
    launch = json.loads((root / "launch.json").read_text())
    report = {"scope": "Observed invocation state only; no performance or predictive accuracy claim",
              "checkpoint_sha256": launch["checkpoint_sha256"], "workloads": [], "sources": {}}
    for case in ("current8k_c1", "current8k_c10_late"):
        directory = root / "capture" / case
        files = sorted(directory.glob("inputs-rank-*.jsonl"))
        if {path.name for path in files} != {f"inputs-rank-{rank}.jsonl" for rank in range(16)}:
            raise ValueError("Missing input capture rank")
        records = []
        for path in files:
            content = path.read_bytes()
            report["sources"][str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
            rows = [json.loads(line) for line in content.splitlines()]
            if len(rows) != 3 or [row["ordinal"] for row in rows] != [0, 1, 2]:
                raise ValueError("Expected exactly three ordered forward observations per rank")
            records.extend(rows)
        summary = {"case": case, "rank_forward_records": len(records), "representative_forwards": []}
        for record in sorted(records, key=lambda row: (row["rank"], row["ordinal"])):
            calls = record["calls"]
            counts = collections.Counter(call["kernel"] for call in calls)
            active_attention = record["execution"] == "graph" or record["batch_size"] > 0
            expected = {"torch_npu.npu_lightning_indexer": 21 if active_attention else 0,
                        "torch_npu.npu_sparse_flash_attention": 78 if active_attention else 0}
            if any(counts[kernel] != count for kernel, count in expected.items()):
                raise ValueError("Incomplete full/shared attention layer observations")
            by_module = {call["module"]: call for call in calls if call.get("module")}
            projections = {}
            for name in ("model.layers.0.self_attn.q_b_proj", "model.layers.0.mlp.gate_up_proj",
                         "model.layers.3.mlp.shared_experts.gate_up_proj"):
                if name in by_module:
                    projections[name] = by_module[name]["arguments"]["input"]["shape"]
            first_indexer = next((call for call in calls if call["kernel"] == "torch_npu.npu_lightning_indexer"), None)
            sequence_lengths = None
            if first_indexer:
                sequence_lengths = first_indexer["arguments"]["actual_seq_lengths_key"]["captured_values"]
                if record["execution"] == "graph":
                    if len(sequence_lengths) != 8 or any(length < 0 for length in sequence_lengths):
                        raise ValueError("Invalid graph sequence metadata")
                    if any(sequence_lengths[record["batch_size"]:]):
                        raise ValueError("Unexpected nonzero padding sequence lengths")
            index_weights = [call for call in calls if call["kernel"] == "torch.nn.functional.linear"
                             and ".indexer.weights_proj.weight" in str(call["arguments"])]
            for call in index_weights:
                arguments = call["arguments"]["positional"]
                if (arguments[0]["dtype"], arguments[1]["dtype"], call["output"]["dtype"]) != (
                    "torch.float32", "torch.bfloat16", "torch.float32"
                ):
                    raise ValueError("Unexpected index weighting precision")
            if record["rank"] in (0, 8):
                summary["representative_forwards"].append({
                    "rank": record["rank"], "ordinal": record["ordinal"], "execution": record["execution"],
                    "logical_batch": record["batch_size"], "global_num_tokens_cpu": record["global_num_tokens_cpu"],
                    "kernel_counts": dict(counts), "projection_inputs": projections,
                    "indexer_sequence_lengths": sequence_lengths,
                })
        responses = list(directory.glob("response*.json"))
        if len(responses) != (1 if case == "current8k_c1" else 10):
            raise ValueError("Incomplete workload responses")
        for path in responses:
            metadata = json.loads(path.read_text())["meta_info"]
            if tuple(metadata[key] for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "num_retractions")) != (8192, 1024, 0, 0):
                raise ValueError("Workload differs from requested shape")
        summary["completed_requests"] = len(responses)
        report["workloads"].append(summary)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "workloads": len(report["workloads"]),
                      "rank_forward_records": sum(row["rank_forward_records"] for row in report["workloads"])}))


if __name__ == "__main__":
    main()
