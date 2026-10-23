"""Publish verified expert chains while retaining explicit routing profiles."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path


def validated_rows(plan_path, observations, runtime_identity):
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    required = (
        "image_digest",
        "framework_commit",
        "driver_version_info",
        "hardware",
        "collector_source_sha256",
        "torch",
        "torch_npu",
    )
    if any(not runtime_identity.get(field) for field in required):
        raise ValueError("Incomplete runtime identity")
    if plan.get("schema") != "aisimulate-ascend-moe-chain-plan/v1":
        raise ValueError("Unsupported MoE chain plan")
    seen, rows, rejected = set(), [], []
    protocol, environment = None, None
    for path in observations:
        source_bytes = path.read_bytes()
        record = json.loads(source_bytes)
        if record.get("schema") != "aisimulate-ascend-moe-chain-observations/v1" or record.get("status") != "completed":
            raise ValueError("Incomplete/unsupported MoE observation")
        if (
            record["plan_sha256"] != plan_hash
            or record["collector_source_sha256"] != runtime_identity["collector_source_sha256"]
        ):
            raise ValueError("Plan/collector source mismatch")
        for field in ("framework_commit", "driver_version_info", "torch", "torch_npu"):
            if record["runtime"][field] != runtime_identity[field]:
                raise ValueError(f"Runtime identity mismatch: {field}")
        if record["runtime"]["device"] != runtime_identity["hardware"]:
            raise ValueError("Runtime hardware mismatch")
        current = record["protocol"]
        if (
            current != plan["protocol"]
            or min(current.values()) < 1
            or current["repeats"] < 3
            or current["iterations"] % current["weight_copies"]
        ):
            raise ValueError("Invalid measurement protocol")
        setup = record["runtime"]["serving_setup"]
        if setup["worker_cpu_affinity"][str(record["runtime"]["device_index"])] != record["runtime"]["cpu_affinity"]:
            raise ValueError("Serving CPU affinity mismatch")
        if protocol is not None and (protocol != current or environment != setup):
            raise ValueError("Cannot merge different protocols or serving setups")
        protocol, environment = current, setup
        if [row["plan_index"] for row in record["cases"]] != record["selected_plan_indices"]:
            raise ValueError("Missing planned shard results")
        for row in record["cases"]:
            index = row["plan_index"]
            if type(index) is not int or not 0 <= index < len(plan["cases"]) or index in seen:
                raise ValueError("Duplicate or unknown case identity")
            seen.add(index)
            case = plan["cases"][index]
            if any(row.get(field) != value for field, value in case.items()):
                raise ValueError("Case differs from the frozen plan")
            experts, count = case["local_experts"], case["received_tokens"]
            if (
                experts not in (8, 16)
                or case["phase"] not in ("prefill", "decode")
                or case["execution_mode"] not in ("eager", "graph")
                or case["hidden_size"] != 6144
                or case["intermediate_size"] != 2048
            ):
                raise ValueError("Unsupported expert geometry")
            capacity = count if case["phase"] == "prefill" else (256 // experts) * 32 * 8
            if not 0 < count <= capacity or case["physical_rows"] != capacity or case["group_list_type"] != 1:
                raise ValueError("Count vectors must fit the declared physical buffer")
            distributions = case["distributions"]
            if not distributions or len(distributions) != len(set(distributions)):
                raise ValueError("Invalid distribution aliases")
            for distribution in distributions:
                if distribution == "balanced":
                    counts = [count // experts + int(index < count % experts) for index in range(experts)]
                elif distribution == "single_expert":
                    counts = [count] + [0] * (experts - 1)
                else:
                    raise ValueError("Unknown expert distribution")
                if case["expert_counts"] != counts:
                    raise ValueError("Expert distribution differs from the actual count vector")
            kernels = {
                "gmm": "sglang.srt.hardware_backend.npu.moe.matmul.GroupedMatmul",
                "activation": "sgl_kernel_npu.activation.swiglu_quant.swiglu_quant",
            }
            verification = "captured_graph_replay" if case["execution_mode"] == "graph" else "eager_invocation"
            if (
                row["weight_format"] != 29
                or row["input_dtype"] != "int8"
                or row["output_dtype"] != "bfloat16"
                or row["scale_dtypes"] != ["float32", "bfloat16"]
                or row["kernel_sources"] != kernels
            ):
                raise ValueError("MoE precision/kernel contract mismatch")
            if (
                row["execution_verification"] != verification
                or len(row["checks"]) != 2
                or row["weight_copies"] != current["weight_copies"]
            ):
                raise ValueError("Actual execution and weight rotation must be checked")
            for check in row["checks"]:
                if (
                    check["sampled_elements"] <= 0
                    or not math.isfinite(check["max_abs_error"])
                    or check["max_abs_error"] < 0
                    or not 0 <= check["max_quant_bin_error"] <= 1
                    or not math.isfinite(check["max_quant_scale_error"])
                    or check["max_quant_scale_error"] < 0
                ):
                    raise ValueError("Invalid numerical/quantization check")
            values = row["samples_ms"]
            if len(values) != current["repeats"] or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("Invalid timing samples")
            variation = statistics.pstdev(values) / statistics.mean(values)
            if variation > 0.1:
                rejected.append({"plan_index": index, "cv": variation, "reason": "CV exceeds10%"})
                continue
            for distribution in distributions:
                result = {name: value for name, value in case.items() if name not in ("distributions", "expert_counts")}
                result.update(
                    distribution=distribution,
                    expert_counts=json.dumps(case["expert_counts"], separators=(",", ":")),
                    weight_format=29,
                    input_dtype="int8",
                    output_dtype="bfloat16",
                    gmm_source=kernels["gmm"],
                    activation_source=kernels["activation"],
                    execution_verification=verification,
                    latency_ms=statistics.median(values),
                    cv=variation,
                    sample_count=len(values),
                    numerical_checks=sum(check["sampled_elements"] for check in row["checks"]),
                    source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                )
                rows.append(result)
    if seen != set(range(len(plan["cases"]))):
        raise ValueError("Publication requires the complete frozen plan")
    if not rows:
        raise ValueError("No qualified measurements")
    identity = {
        "runtime": runtime_identity,
        "protocol": protocol,
        "serving_setup": environment,
        "plan_sha256": plan_hash,
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for row in rows:
        row["runtime_identity_sha256"] = identity_hash
    return rows, {
        "identity": identity,
        "attempted_physical_cases": len(seen),
        "qualified_physical_cases": len(seen) - len(rejected),
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
        hashlib.sha256(Path(__file__).with_name("ascend_moe_chain.py").read_bytes()).hexdigest()
        != identity["collector_source_sha256"]
    ):
        raise ValueError("Publish from the exact measured collector source")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parquet.write_table(
        arrow.Table.from_pylist(rows), args.output_dir / "ascend_moe_chain_perf.parquet", compression="zstd"
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
            "abi": {name: identity[name] for name in ("torch", "torch_npu", "driver_version_info")},
        },
        {
            "ascend_moe_chain_perf": {
                "collector_ref": args.collector_ref,
                "collector_hash": collector_hash(
                    "collector.sglang.ascend_moe_chain",
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
    print(
        json.dumps(
            {
                name: report[name]
                for name in ("attempted_physical_cases", "qualified_physical_cases", "published_profile_rows")
            }
        )
    )


if __name__ == "__main__":
    main()
