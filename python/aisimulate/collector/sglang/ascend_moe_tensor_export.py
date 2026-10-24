"""Publish complete, numerically checked Ascend tensor-MoE campaigns."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

KERNEL_SOURCES = [
    "sglang.srt.hardware_backend.npu.moe.init_routing.NPUMoEInitRouting_v2",
    "sglang.srt.hardware_backend.npu.quantization.moe_methods.NPUW8A8Int8MoEMethod",
    "sglang.srt.hardware_backend.npu.moe.activation.NPUSwigluQuant",
    "sglang.srt.hardware_backend.npu.moe.finalize_routing.NPUFinalizeRouting",
]
CONTRACT = "ascend_tp_w8a8/v1"


def validated_rows(plan_path, observations, runtime_identity):
    plan = json.loads(plan_path.read_text())
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    required = (
        "image_digest",
        "framework_commit",
        "driver_version_info",
        "hardware",
        "collector_source_sha256",
        "torch",
        "torch_npu",
    )
    if plan.get("schema") != "aisimulate-ascend-moe-tensor-plan/v1" or any(
        not runtime_identity.get(field) for field in required
    ):
        raise ValueError("Missing tensor-MoE plan/runtime identity")
    protocol = plan["protocol"]
    if min(protocol.values()) < 1 or protocol["repeats"] < 3 or protocol["iterations"] % protocol["weight_copies"]:
        raise ValueError("Invalid tensor-MoE protocol")
    seen, rows, rejected, setup = set(), [], [], None
    for path in observations:
        payload = path.read_bytes()
        record = json.loads(payload)
        if (
            record.get("schema") != "aisimulate-ascend-moe-tensor-observations/v1"
            or record.get("status") != "completed"
            or record.get("plan_sha256") != plan_hash
            or record.get("collector_source_sha256") != runtime_identity["collector_source_sha256"]
            or record.get("protocol") != protocol
        ):
            raise ValueError("Incomplete or mismatched tensor-MoE observation")
        runtime = record["runtime"]
        if any(
            runtime[field] != runtime_identity[field]
            for field in ("framework_commit", "torch", "torch_npu", "driver_version_info")
        ):
            raise ValueError("Runtime identity mismatch")
        if runtime["device"] != runtime_identity["hardware"]:
            raise ValueError("Hardware identity mismatch")
        current_setup = runtime["serving_setup"]
        if current_setup["worker_cpu_affinity"][str(runtime["device_index"])] != runtime["cpu_affinity"]:
            raise ValueError("Serving CPU placement mismatch")
        if setup is not None and setup != current_setup:
            raise ValueError("Different serving setups cannot share a tensor-MoE table")
        setup = current_setup
        if [row["plan_index"] for row in record["cases"]] != record["selected_plan_indices"]:
            raise ValueError("Missing planned tensor-MoE results")
        for row in record["cases"]:
            index = row["plan_index"]
            if type(index) is not int or index in seen or not 0 <= index < len(plan["cases"]):
                raise ValueError("Unknown or duplicate physical case")
            seen.add(index)
            case = plan["cases"][index]
            if any(row.get(field) != value for field, value in case.items()):
                raise ValueError("Observed tensor-MoE case differs from plan")
            if (
                (case["hidden_size"], case["num_experts"], case["top_k"]) != (6144, 256, 8)
                or case["intermediate_size"] not in (64, 128, 256, 512)
                or case["distribution"] != "balanced"
                or case["execution_mode"] not in ("eager", "graph")
                or case["tokens"] < 1
            ):
                raise ValueError("Unsupported tensor-MoE geometry")
            verification = "captured_graph_replay" if case["execution_mode"] == "graph" else "eager_invocation"
            expected = {
                "status": "passed",
                "weight_format": 29,
                "input_dtype": "bfloat16",
                "output_dtype": "bfloat16",
                "group_list_type": 1,
                "quant_mode": 1,
                "drop_pad_mode": 2,
                "weight_copies": protocol["weight_copies"],
                "kernel_sources": KERNEL_SOURCES,
                "execution_verification": verification,
            }
            if any(row.get(field) != value for field, value in expected.items()):
                raise ValueError("Tensor-MoE kernel or precision contract mismatch")
            checks = row["checks"]
            if len(checks) != 2 or [check.get("input_mutation") for check in checks] != [-1, 1]:
                raise ValueError("Both input mutations must verify actual execution")
            for check in checks:
                if (
                    check["sampled_elements"] < 1
                    or not math.isfinite(check["max_abs_error"])
                    or check["max_abs_error"] < 0
                ):
                    raise ValueError("Invalid tensor-MoE numerical check")
            values = row["samples_ms"]
            wall_values = row["wall_samples_ms"]
            if (
                len(values) != protocol["repeats"]
                or len(wall_values) != len(values)
                or any(not math.isfinite(value) or value <= 0 for value in [*values, *wall_values])
            ):
                raise ValueError("Invalid tensor-MoE timing samples")
            variation = statistics.pstdev(values) / statistics.mean(values)
            if variation > 0.1:
                rejected.append({"plan_index": index, "reason": "CV exceeds10%", "cv": variation})
                continue
            rows.append(
                {
                    **case,
                    "kernel_contract": CONTRACT,
                    "weight_format": 29,
                    "input_dtype": "bfloat16",
                    "output_dtype": "bfloat16",
                    "group_list_type": 1,
                    "quant_mode": 1,
                    "drop_pad_mode": 2,
                    "kernel_sources": json.dumps(KERNEL_SOURCES, separators=(",", ":")),
                    "execution_verification": verification,
                    "latency_ms": statistics.median(values),
                    "cv": variation,
                    "sample_count": len(values),
                    "numerical_checks": sum(check["sampled_elements"] for check in checks),
                    "source_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    if seen != set(range(len(plan["cases"]))):
        raise ValueError("Publication requires the complete frozen plan")
    if not rows:
        raise ValueError("No qualified tensor-MoE measurements")
    identity = {"runtime": runtime_identity, "protocol": protocol, "serving_setup": setup, "plan_sha256": plan_hash}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for row in rows:
        row["runtime_identity_sha256"] = identity_hash
    return rows, {
        "identity": identity,
        "runtime_identity_sha256": identity_hash,
        "attempted_physical_cases": len(seen),
        "published_profile_rows": len(rows),
        "unqualified": rejected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, nargs="+")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collector-ref", required=True)
    args = parser.parse_args()
    rows, report = validated_rows(args.plan, args.observations, json.loads(args.runtime_identity.read_text()))
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    from collector.provenance import case_plan_hash, collector_hash, load_closures, write_collection_meta

    package = Path(__file__).resolve().parents[2]
    identity = report["identity"]["runtime"]
    if (
        hashlib.sha256(Path(__file__).with_name("ascend_moe_tensor.py").read_bytes()).hexdigest()
        != identity["collector_source_sha256"]
    ):
        raise ValueError("Publish from the exact measured collector source")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parquet.write_table(
        arrow.Table.from_pylist(rows), args.output_dir / "ascend_moe_tensor_perf.parquet", compression="zstd"
    )
    (args.output_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    plan = json.loads(args.plan.read_text())
    write_collection_meta(
        args.output_dir,
        {
            "framework": "sglang",
            "version": "0.5.22",
            "image_digest": identity["image_digest"],
            "source_commit": identity["framework_commit"],
            "abi": {field: identity[field] for field in ("torch", "torch_npu", "driver_version_info")},
        },
        {
            "ascend_moe_tensor_perf": {
                "collector_ref": args.collector_ref,
                "collector_hash": collector_hash(
                    "collector.sglang.ascend_moe_tensor",
                    package,
                    load_closures(package / "collector/hash_closures.yaml"),
                ),
                "case_plan_hash": case_plan_hash(
                    [json.dumps(case, sort_keys=True, separators=(",", ":")) for case in plan["cases"]]
                ),
                "collected_at": datetime.fromtimestamp(
                    max(path.stat().st_mtime for path in args.observations), UTC
                ).isoformat(),
                "rows": len(rows),
                "classified_failures": len(report["unqualified"]),
                "status": "complete",
            }
        },
        provenance_tier="collected",
    )
    print(json.dumps({"attempted": report["attempted_physical_cases"], "published": len(rows)}))


if __name__ == "__main__":
    main()
