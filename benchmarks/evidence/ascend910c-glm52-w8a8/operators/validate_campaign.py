"""Check this campaign's workload, exact operator counts and retained evidence."""

import hashlib
import argparse
import json
import urllib.request
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="repeat-20261021")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    campaign = root / args.campaign
    manifest = load(campaign / "manifest.json")
    report = load(campaign / "trace-report.json")
    counts = load(campaign / "kernel-counts.json")
    groups = load(campaign / "kernel-groups.json")
    before = load(campaign / "server-info-before.json")
    after = load(campaign / "server-info-after.json")
    topology = {
        "tp_size": 16,
        "attn_dp_size": 2,
        "ep_size": 16,
        "quantization": "modelslim",
        "dtype": "bfloat16",
    }
    for key, expected in topology.items():
        require(
            before[key] == after[key] == expected,
            f"Serving configuration changed: {key}",
        )
    control = load(campaign / "control/response.json")
    require(len(control["output_ids"]) == 1024, "Control output length differs")
    profiles = [
        case for case in manifest["cases"] if case["path"].startswith("profile-")
    ]
    require(len(profiles) == 3, "Expected three independent repetitions")
    expected_counts = {
        ("eager", "GroupedMatmul"): 150,
        ("eager", "CamMoeDispatchNormal"): 75,
        ("eager", "CamMoeCombineNormal"): 75,
        ("graph", "GroupedMatmul"): 300,
        ("graph", "SparseFlashAttention"): 156,
        ("graph", "LightningIndexer"): 42,
        ("graph", "MoeLowLatencyDispatchV2"): 150,
        ("graph", "MoeLowLatencyCombineV2"): 150,
    }
    verified_counts = 0
    for case in manifest["cases"]:
        metadata = case["meta_info"]
        for field, expected in {
            "prompt_tokens": 8192,
            "completion_tokens": 1024,
            "cached_tokens": 0,
            "num_retractions": 0,
            "dp_rank": 0,
        }.items():
            require(metadata[field] == expected, f"{case['path']}: incorrect {field}")
        response = load(campaign / case["path"] / "response.json")
        require(
            response["output_ids"] == control["output_ids"],
            f"Output tokens differ: {case['path']}",
        )
        if case["path"] == "control":
            continue
        selected = {
            (int(row["device"]), row["execution"], row["type"]): row["count"]
            for row in counts
            if row["case"] == case["path"]
        }
        for rank in range(16):
            for (execution, operation), count in expected_counts.items():
                require(
                    selected.get((rank, execution, operation), 0) == count,
                    f"{case['path']}: {rank}/{operation} count differs",
                )
                verified_counts += 1
            for operation, count in [
                ("LightningIndexer", 21),
                ("SparseFlashAttention", 78),
            ]:
                expected = count if rank < 8 else 0
                require(
                    selected.get((rank, "eager", operation), 0) == expected,
                    f"{case['path']}: eager DP boundary differs",
                )
                verified_counts += 1
    hashes = load(campaign / "sha256.json")
    for relative, expected in hashes.items():
        path = campaign / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        require(
            digest.hexdigest() == expected,
            f"Original collection artifact changed: {relative}",
        )
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=30) as response:
        health = response.status
    families = {}
    for operation in sorted({row["Type"] for row in groups}):
        selected = [row for row in groups if row["Type"] == operation]
        families[operation] = {
            "groups": len(selected),
            "repeatable_groups": sum(
                row["repeatable_at_10_percent"] for row in selected
            ),
        }
    retained_hashes = {}
    for path in sorted(campaign.rglob("*")):
        if path.is_file() and path.name not in {
            "trace.tar.gz",
            "collection-validation.json",
        }:
            retained_hashes[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    result = {
        "collection_integrity": "passed",
        "workloads_passed": 4,
        "output_ids_identical": True,
        "ranks_per_profile": 16,
        "profiled_forwards_per_rank": 3,
        "operator_count_assertions": verified_counts,
        "kernel_records": sum(run["kernel_records"] for run in report["runs"]),
        "repeatable_groups": report["repeatable_groups"],
        "unqualified_groups": report["unqualified_groups"],
        "families": families,
        "service_health": health,
        "retained_artifact_sha256": retained_hashes,
        "qualification": "Repeated real-serving observations only; isolated numeric correctness and production FPM lookup remain incomplete.",
        "pilot_failure": "Probe inference and traces succeeded; redundant stop_profile after automatic stop returned500. Fixed collector verifies completed exports and retains pilot failure evidence.",
    }
    (campaign / "collection-validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"families", "retained_artifact_sha256"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
