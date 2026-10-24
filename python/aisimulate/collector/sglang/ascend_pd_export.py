"""Validate complete Ascend replica-transfer campaigns and publish bounded rows."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path


def validated_rows(plan_path, observations, runtime_identity):
    payload = plan_path.read_bytes()
    plan = json.loads(payload)
    plan_hash = hashlib.sha256(payload).hexdigest()
    fields = ("torch", "torch_npu", "memfabric", "hardware", "driver")
    if any(
        not runtime_identity.get(name)
        for name in (*fields, "framework_commit", "image_digest", "collector_source_sha256")
    ):
        raise ValueError("Incomplete PD runtime identity")
    if (
        plan.get("schema") != "aisimulate-ascend-pd-plan/v1"
        or plan["protocol"]["repeats"] < 3
        or any(type(value) is not int or value < 1 for value in plan["protocol"].values())
    ):
        raise ValueError("Invalid PD plan")
    seen, rows, rejected, placements = set(), [], [], []
    setup = None
    for path in observations:
        raw = path.read_bytes()
        record = json.loads(raw)
        if (
            record.get("schema") != "aisimulate-ascend-pd-observations/v1"
            or record.get("status") != "completed"
            or record.get("negative_control")
        ):
            raise ValueError("Incomplete or negative PD campaign")
        if (
            record["plan_sha256"] != plan_hash
            or record["collector_source_sha256"] != runtime_identity["collector_source_sha256"]
            or record["framework_commit"] != runtime_identity["framework_commit"]
        ):
            raise ValueError("PD source/plan identity mismatch")
        if record["protocol"] != plan["protocol"] or (setup is not None and setup != record["serving_setup"]):
            raise ValueError("PD campaigns have different protocols or setup")
        setup = record["serving_setup"]
        devices = record["devices"]
        if len(devices) != 32 or [device["rank"] for device in devices] != list(range(32)):
            raise ValueError("Incomplete PD rank inventory")
        for device in devices:
            if any(device[name] != runtime_identity[name] for name in fields):
                raise ValueError("PD rank runtime mismatch")
            if (
                device["source_sha256"] != record["collector_source_sha256"]
                or device["plan_sha256"] != plan_hash
                or device["selected_plan_indices"] != record["selected_plan_indices"]
            ):
                raise ValueError("PD rank source/selection mismatch")
            if (
                device["local_rank"] != device["rank"] % 16
                or device["cpu_affinity"] != setup["worker_cpu_affinity"][str(device["local_rank"])]
            ):
                raise ValueError("PD rank layout or affinity mismatch")
            if device["pool_shapes"] != [[78, 417, 128, 1, width] for width in (512, 64, 128)] or device[
                "pool_formats"
            ] != [30, 30, 30]:
                raise ValueError("PD memory-pool layout mismatch")
        if devices[0]["host"] == devices[16]["host"] or any(
            device["host"] != devices[device["rank"] // 16 * 16]["host"] for device in devices
        ):
            raise ValueError("Expected one prefill and one decode node")
        placements.append({"source": devices[0]["host"], "destination": devices[16]["host"]})
        if [case["plan_index"] for case in record["cases"]] != record["selected_plan_indices"]:
            raise ValueError("Missing selected PD work")
        for observed in record["cases"]:
            index = observed["plan_index"]
            if type(index) is not int or not 0 <= index < len(plan["cases"]) or index in seen:
                raise ValueError("Duplicate or unknown PD case")
            seen.add(index)
            case = plan["cases"][index]
            if any(
                device["source_tp"] != case["source_tp"] or device["destination_tp"] != case["destination_tp"]
                for device in devices
            ):
                raise ValueError("PD participant topology differs from the selected case")
            if any(observed.get(name) != value for name, value in case.items()):
                raise ValueError("PD case differs from frozen plan")
            if (
                case["source_tp"] not in (1, 2, 4, 8, 16)
                or case["destination_tp"] not in (1, 2, 4, 8, 16)
                or case["ranks_per_node"] != 16
                or case["num_layers"] != 78
                or case["page_size"] != 128
                or case["reference_pool_tokens"] != 53248
                or case["component_widths"] != [512, 64, 128]
                or case["dtype"] != "bfloat16"
                or case["protocol"] != "sdma"
                or case["active_requests"] != 1
                or case["tokens"] <= 0
                or case["tokens"] % 128
            ):
                raise ValueError("Unsupported PD geometry")
            expected_mapping = [
                {
                    "source_rank": destination * case["source_tp"] // case["destination_tp"],
                    "destination_rank": 16 + destination,
                }
                for destination in range(case["destination_tp"])
            ]
            if record["rank_mapping"] != expected_mapping:
                raise ValueError("PD mapping differs from the actual MLA placement")
            count = case["tokens"] // 128
            source_pages, destination_pages = observed["source_pages"], observed["destination_pages"]
            for pages in (source_pages, destination_pages):
                if (
                    len(pages) != count
                    or len(set(pages)) != count
                    or any(type(page) is not int or not 1 <= page <= 416 for page in pages)
                ):
                    raise ValueError("Invalid PD page indices")
            segments = 1 + sum(
                source_pages[position] != source_pages[position - 1] + 1
                or destination_pages[position] != destination_pages[position - 1] + 1
                for position in range(1, count)
            )
            if observed["segments_per_buffer"] != segments:
                raise ValueError("PD coalesced-run count differs")
            if case["page_layout"] == "contiguous":
                if source_pages != list(range(1, count + 1)) or destination_pages != list(
                    range(count + 1, 2 * count + 1)
                ):
                    raise ValueError("PD contiguous page placement differs")
            elif case["page_layout"] == "fragmented":
                if segments != count:
                    raise ValueError("Fragmented profile requires one transfer run per page")
            else:
                raise ValueError("Unknown PD page layout")
            replica_bytes = case["tokens"] * 109824
            if (
                observed["payload_bytes_per_replica"] != replica_bytes
                or observed["total_wire_bytes"] != replica_bytes * case["destination_tp"]
            ):
                raise ValueError("PD payload bytes differ from the actual BF16 layout")
            results = observed["rank_results"]
            if [result["rank"] for result in results] != list(range(32)):
                raise ValueError("Missing PD rank result")
            digests = [None, None]
            for result in results:
                destinations = [
                    entry["destination_rank"] for entry in expected_mapping if entry["source_rank"] == result["rank"]
                ]
                if result["destinations"] != destinations or len(result["samples_ms"]) != plan["protocol"]["repeats"]:
                    raise ValueError("PD transfer execution differs from placement")
                if any(
                    not math.isfinite(value) or value < 0 or (bool(value) != bool(destinations))
                    for value in result["samples_ms"]
                ):
                    raise ValueError("Invalid PD timing sample")
                iterations = plan["protocol"].get("iterations", 1)
                calls = result.get("call_samples_ms")
                if calls is None and iterations != 1:
                    raise ValueError("Averaged PD timing requires all per-call observations")
                if calls is not None:
                    if len(calls) != plan["protocol"]["repeats"] or any(len(values) != iterations for values in calls):
                        raise ValueError("PD timing call count differs from protocol")
                    for values, average in zip(calls, result["samples_ms"], strict=True):
                        if any(
                            not math.isfinite(value) or value < 0 or bool(value) != bool(destinations)
                            for value in values
                        ):
                            raise ValueError("Invalid PD per-call timing")
                        if statistics.mean(values) != average:
                            raise ValueError("PD timing must be the per-call mean")
                if len(result["checks"]) != 2:
                    raise ValueError("PD requires before/after full-payload checks")
                receiver = 16 <= result["rank"] < 16 + case["destination_tp"]
                for check_index, check in enumerate(result["checks"]):
                    digest = check["sha256"]
                    if len(digest) != 64 or any(letter not in "0123456789abcdef" for letter in digest):
                        raise ValueError("Invalid PD payload digest")
                    if digests[check_index] is None:
                        digests[check_index] = digest
                    if digest != digests[check_index] or check["checked_bytes"] != (replica_bytes if receiver else 0):
                        raise ValueError("PD payload verification differs across replicas")
            if digests[0] == digests[1]:
                raise ValueError("PD payload was not changed between checks")
            critical = [
                max(result["samples_ms"][sample] for result in results) for sample in range(plan["protocol"]["repeats"])
            ]
            latency = statistics.median(critical)
            variation = statistics.pstdev(critical) / statistics.mean(critical)
            if observed["latency_ms"] != latency or observed["cv"] != variation:
                raise ValueError("PD aggregate differs from rank timings")
            if variation > 0.1:
                rejected.append({"plan_index": index, "cv": variation})
                continue
            if observed["kernel_source"] != "sglang.srt.disaggregation.ascend.conn.AscendKVManager.send_kvcache":
                raise ValueError("Unqualified PD transfer implementation")
            row = {name: value for name, value in case.items() if name != "component_widths"}
            row.update(
                component_widths="[512,64,128]",
                latency_ms=latency,
                cv=variation,
                sample_count=plan["protocol"]["repeats"],
                payload_bytes_per_replica=replica_bytes,
                total_wire_bytes=observed["total_wire_bytes"],
                segments_per_buffer=segments,
                verified_receiver_copies=2 * case["destination_tp"],
                execution_verification="full_payload_sha256_before_after",
                kernel_source=observed["kernel_source"],
                source_sha256=hashlib.sha256(raw).hexdigest(),
            )
            rows.append(row)
    if seen != set(range(len(plan["cases"]))) or not rows:
        raise ValueError("PD publication requires the complete frozen plan and qualified rows")
    identity = {
        "runtime": runtime_identity,
        "protocol": plan["protocol"],
        "serving_setup": setup,
        "plan_sha256": plan_hash,
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for row in rows:
        row["runtime_identity_sha256"] = identity_hash
    return rows, {
        "identity": identity,
        "placements": placements,
        "attempted": len(seen),
        "qualified": len(rows),
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
    identity = json.loads(args.runtime_identity.read_text())
    rows, report = validated_rows(args.plan, args.observations, identity)
    if (
        hashlib.sha256(Path(__file__).with_name("ascend_pd.py").read_bytes()).hexdigest()
        != identity["collector_source_sha256"]
    ):
        raise ValueError("Publish from the exact measured PD source")
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    from collector.provenance import case_plan_hash, collector_hash, load_closures, write_collection_meta

    package = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "metadata").mkdir()
    parquet.write_table(arrow.Table.from_pylist(rows), args.output_dir / "ascend_pd_perf.parquet", compression="zstd")
    (args.output_dir / "metadata/provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    plan = json.loads(args.plan.read_text())
    write_collection_meta(
        args.output_dir,
        {
            "framework": "sglang",
            "version": "0.5.22",
            "image_digest": identity["image_digest"],
            "source_commit": identity["framework_commit"],
            "abi": {name: identity[name] for name in ("torch", "torch_npu", "memfabric", "driver")},
        },
        {
            "ascend_pd_perf": {
                "collector_ref": args.collector_ref,
                "collector_hash": collector_hash(
                    "collector.sglang.ascend_pd", package, load_closures(package / "collector/hash_closures.yaml")
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
    print(json.dumps({"qualified": len(rows), "unqualified": len(report["unqualified"])}))


if __name__ == "__main__":
    main()
