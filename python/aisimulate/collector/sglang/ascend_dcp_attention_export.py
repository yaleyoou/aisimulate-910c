"""Publish complete DCP campaigns with rank-wise numerical and timing evidence."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

CONTRACT = "sglang_npu_dcp_storage_attention_chain/v1"


def observation_bytes(path):
    if path.is_file():
        return path.read_bytes()
    return gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())


def validated_rows(plan_path, campaigns, runtime_identity):
    plan = json.loads(plan_path.read_text())
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if plan.get("schema") != "aisimulate-ascend-dcp-attention-plan/v1" or plan.get("boundary_contract") != CONTRACT:
        raise ValueError("Unsupported DCP plan or operation boundary")
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
        raise ValueError("Missing DCP runtime identity")
    protocol = plan["protocol"]
    if any(type(value) is not int or value <= 0 for value in protocol.values()) or protocol["repeats"] < 3:
        raise ValueError("Invalid DCP timing protocol")
    seen, rows, rejected, sources = set(), [], [], []
    setup = None
    for directory in campaigns:
        manifest = json.loads((directory / "campaign.json").read_text())
        if (
            manifest.get("schema") != "aisimulate-ascend-dcp-campaign/v1"
            or manifest.get("status") not in ("completed", "failed")
            or manifest.get("negative_control")
            or manifest.get("contamination")
            or not manifest.get("attempts")
        ):
            raise ValueError("DCP publication requires a completed uncontaminated full campaign")
        stopped_for_investigation = manifest["status"] == "failed"
        if stopped_for_investigation and (
            "DCP numerical failures reached the investigation threshold" not in manifest.get("error", "")
            or not manifest.get("failed_cases")
        ):
            raise ValueError("Only an audited numerical-threshold stop can contribute a DCP prefix")
        if stopped_for_investigation:
            audit_path = directory / "investigation.json"
            if not audit_path.is_file():
                raise ValueError("Stopped DCP prefix requires its recorded investigation")
            audit = json.loads(audit_path.read_text())
            if audit.get("status") != "audited_numerical_exclusions" or sorted(
                audit.get("affected_indices", [])
            ) != sorted(entry["plan_index"] for entry in manifest["failed_cases"]):
                raise ValueError("DCP investigation does not cover the excluded cases")
        planned = manifest["planned_indices"]
        local_seen = set()
        failed = {entry["plan_index"]: entry for entry in manifest["failed_cases"]}
        if len(failed) != len(manifest["failed_cases"]):
            raise ValueError("Duplicate failed DCP case")
        for attempt in manifest["attempts"]:
            if Path(attempt["observation"]).name != attempt["observation"]:
                raise ValueError("DCP observation must be inside its campaign")
            path = directory / attempt["observation"]
            payload = observation_bytes(path)
            record = json.loads(payload)
            if (
                record.get("schema") != "aisimulate-ascend-dcp-attention-observations/v1"
                or record.get("status") not in ("completed", "failed")
                or record.get("plan_sha256") != plan_hash
                or record.get("collector_source_sha256") != runtime_identity["collector_source_sha256"]
                or record.get("framework_commit") != runtime_identity["framework_commit"]
                or record.get("protocol") != protocol
                or record.get("negative_control")
            ):
                raise ValueError("DCP observation identity mismatch")
            current_setup = record["serving_setup"]
            if setup is not None and setup != current_setup:
                raise ValueError("DCP serving setups differ")
            setup = current_setup
            workers = record["runtime"]
            world = len(workers)
            if world not in (2, 4, 8, 16) or [worker["rank"] for worker in workers] != list(range(world)):
                raise ValueError("Incomplete contiguous DCP rank group")
            if len({worker["host"] for worker in workers}) != 1:
                raise ValueError("DCP group must stay within one node")
            for worker in workers:
                if (
                    worker["local_rank"] != worker["rank"]
                    or worker["device"] != runtime_identity["hardware"]
                    or any(
                        worker[field] != runtime_identity[field]
                        for field in ("torch", "torch_npu", "driver_version_info")
                    )
                    or worker["cpu_affinity"] != setup["worker_cpu_affinity"][str(worker["local_rank"])]
                ):
                    raise ValueError("DCP worker runtime or CPU placement mismatch")
            observed_indices = [row["plan_index"] for row in record["cases"]]
            if (
                observed_indices != attempt["completed_indices"]
                or observed_indices != record["selected_plan_indices"][: len(observed_indices)]
            ):
                raise ValueError("DCP attempt lost or reordered planned results")
            for row in record["cases"]:
                index = row["plan_index"]
                if type(index) is not int or index in seen or index not in planned:
                    raise ValueError("Unknown or repeated DCP point")
                case = plan["cases"][index]
                if any(row.get(field) != value for field, value in case.items()) or case["dcp_size"] != world:
                    raise ValueError("DCP invocation differs from its plan")
                if (
                    case["kv_interleave_size"] != 128
                    or case["page_size"] != 128
                    or case["kv_lora_rank"] != 512
                    or case["rope_dim"] != 64
                    or case["sparse_topk"] != 2048
                    or case["local_heads"] not in (4, 8, 16)
                    or case["local_heads"] * world > 64
                    or type(case["remap_topk"]) is not bool
                    or (case["phase"], case["execution_mode"]) not in (("prefill", "eager"), ("decode", "graph"))
                ):
                    raise ValueError("Unqualified DCP geometry or NPU page ownership")
                expected = {
                    "status": "passed",
                    "kernel_contract": CONTRACT,
                    "cache_format": 30,
                    "query_dtype": "bfloat16",
                    "output_dtype": "bfloat16",
                    "merge_dtype": "float32",
                    "group_ranks": list(range(world)),
                    "rank_layout": "contiguous",
                    "ranks_per_node": world,
                    "local_cache_index_dtype": "int32" if case["execution_mode"] == "graph" else "int64",
                    "indexer_cache_index_dtype": "int64" if case["execution_mode"] == "graph" else "int32",
                    "execution_verification": "captured_graph_replay"
                    if case["execution_mode"] == "graph"
                    else "eager_invocation",
                }
                if any(row.get(field) != value for field, value in expected.items()):
                    raise ValueError("DCP measured boundary or dtype contract mismatch")
                ranks = row["rank_measurements"]
                if len(ranks) != world or [worker["rank"] for worker in ranks] != list(range(world)):
                    raise ValueError("Missing DCP rank measurements")
                numerical_checks = 0
                for worker in ranks:
                    for field in ("samples_ms", "wall_samples_ms"):
                        if len(worker[field]) != protocol["repeats"] or any(
                            not math.isfinite(value) or value <= 0 for value in worker[field]
                        ):
                            raise ValueError("Invalid DCP timing samples")
                    checks = worker["checks"]
                    if len(checks) != 2 or [check["input_mutation"] for check in checks] != [-1, 1]:
                        raise ValueError("DCP must verify both input mutations on every rank")
                    for check in checks:
                        if (
                            check["sampled_elements"] <= 0
                            or check["rank"] != worker["rank"]
                            or any(
                                not math.isfinite(check[field]) or check[field] < 0
                                for field in ("max_abs_error", "max_lse_error", "ideal_fp32_max_abs_error")
                            )
                        ):
                            raise ValueError("Invalid DCP numerical evidence")
                        numerical_checks += check["sampled_elements"]
                values = [
                    max(worker["samples_ms"][repeat] for worker in ranks) for repeat in range(protocol["repeats"])
                ]
                variation = max(
                    [
                        statistics.pstdev(values) / statistics.mean(values),
                        *[
                            statistics.pstdev(worker["samples_ms"]) / statistics.mean(worker["samples_ms"])
                            for worker in ranks
                        ],
                    ]
                )
                if values != row["samples_ms"] or not math.isclose(row["cv"], variation, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError("DCP rank maximum or variation aggregation was altered")
                seen.add(index)
                local_seen.add(index)
                if variation > 0.1:
                    rejected.append({"plan_index": index, "reason": "CV exceeds10%", "cv": variation})
                    continue
                rows.append(
                    {
                        **case,
                        **{field: value for field, value in expected.items() if field not in ("status", "group_ranks")},
                        "latency_ms": statistics.median(values),
                        "cv": variation,
                        "sample_count": len(values),
                        "numerical_checks": numerical_checks,
                        "source_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            if record["status"] == "completed":
                if observed_indices != record["selected_plan_indices"] or attempt["returncode"] != 0:
                    raise ValueError("DCP worker falsely reported complete coverage")
            else:
                index = record["failed_plan_index"]
                remaining = record["selected_plan_indices"][len(observed_indices) :]
                if not remaining or index != remaining[0] or index in seen or index not in failed:
                    raise ValueError("DCP failed case is not the next untouched point")
                failure = failed[index]
                if failure["classification"] != "numerical_check_failed" or failure["observation"] != path.name:
                    raise ValueError("Unreviewed DCP failure classification")
                if any(Path(name).name != name for name in failure["failure_files"]):
                    raise ValueError("DCP failure evidence must stay inside its campaign")
                evidence = [json.loads(observation_bytes(directory / name)) for name in failure["failure_files"]]
                if not any(
                    item["plan_index"] == index
                    and item["case"] == plan["cases"][index]
                    and item["collector_source_sha256"] == runtime_identity["collector_source_sha256"]
                    and "DCP execution numerical check failed" in item["error"]
                    for item in evidence
                ):
                    raise ValueError("DCP numerical exclusion lacks its original worker evidence")
                rejected.append(
                    {
                        "plan_index": index,
                        "reason": "numerical_check_failed",
                        "campaign": str(directory),
                        "failure_files": failure["failure_files"],
                    }
                )
                seen.add(index)
                local_seen.add(index)
            sources.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "runtime": workers})
        expected_local = set(planned[: len(local_seen)]) if stopped_for_investigation else set(planned)
        if local_seen != expected_local or len(local_seen) != manifest["attempted_cases"]:
            raise ValueError("DCP campaign did not attempt its complete frozen selection")
    if seen != set(range(len(plan["cases"]))) or not rows:
        raise ValueError("DCP publication requires every planned case to be attempted")
    identity = {
        "runtime": runtime_identity,
        "protocol": protocol,
        "serving_setup": setup,
        "plan_sha256": plan_hash,
        "kernel_contract": CONTRACT,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for row in rows:
        row["runtime_identity_sha256"] = digest
    return rows, {
        "identity": identity,
        "runtime_identity_sha256": digest,
        "attempted": len(seen),
        "qualified": len(rows),
        "unqualified": rejected,
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, nargs="+")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collector-ref", required=True)
    args = parser.parse_args()
    rows, report = validated_rows(args.plan, args.campaign, json.loads(args.runtime_identity.read_text()))
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    from collector.provenance import case_plan_hash, collector_hash, load_closures, write_collection_meta

    package = Path(__file__).resolve().parents[2]
    identity = report["identity"]["runtime"]
    if (
        hashlib.sha256(Path(__file__).with_name("ascend_dcp_attention.py").read_bytes()).hexdigest()
        != identity["collector_source_sha256"]
    ):
        raise ValueError("Publish from the exact measured DCP collector source")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parquet.write_table(
        arrow.Table.from_pylist(rows), args.output_dir / "ascend_dcp_attention_perf.parquet", compression="zstd"
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
            "ascend_dcp_attention_perf": {
                "collector_ref": args.collector_ref,
                "collector_hash": collector_hash(
                    "collector.sglang.ascend_dcp_attention",
                    package,
                    load_closures(package / "collector/hash_closures.yaml"),
                ),
                "case_plan_hash": case_plan_hash(
                    [json.dumps(case, sort_keys=True, separators=(",", ":")) for case in plan["cases"]]
                ),
                "collected_at": datetime.fromtimestamp(
                    max((path / "campaign.json").stat().st_mtime for path in args.campaign), UTC
                ).isoformat(),
                "rows": len(rows),
                "classified_failures": len(report["unqualified"]),
                "status": "complete",
            }
        },
        provenance_tier="collected",
    )
    print(
        json.dumps({"attempted": report["attempted"], "qualified": len(rows), "excluded": len(report["unqualified"])})
    )


if __name__ == "__main__":
    main()
