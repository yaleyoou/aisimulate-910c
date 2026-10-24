import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import platform
import subprocess
import urllib.request
from pathlib import Path


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reuse-builds", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    skill = args.repo / "python/aisimulate/.agents/skills/aisimulate-replay-parity"
    environment = os.environ.copy()
    environment.pop("CARGO_TARGET_DIR", None)
    environment["CARGO_BUILD_JOBS"] = "16"
    environment.update(
        {
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": str(args.repo),
            "GIT_CONFIG_KEY_1": "safe.directory",
            "GIT_CONFIG_VALUE_1": str(args.repo / ".git"),
        }
    )
    metadata = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "platform": platform.platform(),
        "host": platform.node(),
        "rustc": subprocess.check_output(["/root/.cargo/bin/rustc", "-Vv"], text=True),
        "features": [],
        "profile": "release",
        "determinism": "CanonicalV1",
        "cpu_placement": "one physical core per row, CPUs160..167; all four processes for a row use the same core",
        "exclusions": [
            "/summary/wall_time_ms",
            "/summary/processed_tokens_per_s",
            "/summary/processed_output_tokens_per_s",
            "/planner/html_report_path",
        ],
    }
    run(
        [
            "git",
            "-c",
            "safe.directory=" + str(args.repo),
            "-C",
            str(args.repo),
            "merge-base",
            "--is-ancestor",
            args.baseline,
            args.candidate,
        ]
    )

    def build(label, revision):
        checkout = args.output / (label + "-checkout")
        with (args.output / (label + "-build.log")).open("w") as log:
            run(
                ["git", "clone", "--shared", "--no-checkout", str(args.repo), str(checkout)],
                env=environment,
                stdout=log,
                stderr=log,
            )
            run(["git", "-C", str(checkout), "checkout", "--detach", revision], stdout=log, stderr=log)
            with (args.output / (label + "-build.json")).open("w") as output:
                run(
                    [
                        "python3",
                        str(skill / "scripts/build_runner.py"),
                        "--checkout",
                        str(checkout),
                        "--output-dir",
                        str(args.output / (label + "-runner")),
                        "--prepare-only",
                    ],
                    env=environment,
                    stdout=output,
                    stderr=log,
                )
            record = json.loads((args.output / (label + "-build.json")).read_text())
            runner = args.output / (label + "-runner")
            record["original_lock_sha256"] = sha256(runner / "Cargo.lock")
            run(
                [
                    "/root/.cargo/bin/cargo",
                    "generate-lockfile",
                    "--offline",
                    "--manifest-path",
                    str(runner / "Cargo.toml"),
                ],
                env=environment,
                stdout=log,
                stderr=log,
            )
            record["resolved_lock_sha256"] = sha256(runner / "Cargo.lock")
            record["lock_refresh_reason"] = (
                "Skill lock predates current core dependencies and fails --locked on both revisions; "
                "identical offline lock refresh, runner Rust source unchanged."
            )
            command = [
                "/root/.cargo/bin/cargo",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                str(runner / "Cargo.toml"),
            ]
            run(command, env={**environment, "AISIMULATE_SOURCE_REVISION": revision}, stdout=log, stderr=log)
            binary = runner / "target/release/aisimulate-replay-parity-runner"
            record.update(
                binary=str(binary),
                binary_sha256=sha256(binary),
                binary_size=binary.stat().st_size,
                embedded_source_revision=revision,
                build_command=command,
            )
            (args.output / (label + "-build.json")).write_text(json.dumps(record, indent=2) + "\n")
        return record

    if args.reuse_builds:
        builds = [
            json.loads((args.reuse_builds / (label + "-build.json")).read_text()) for label in ("baseline", "candidate")
        ]
        for record, revision in zip(builds, (args.baseline, args.candidate), strict=True):
            assert record["embedded_source_revision"] == revision
            assert sha256(Path(record["binary"])) == record["binary_sha256"]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(build, label, revision)
                for label, revision in (("baseline", args.baseline), ("candidate", args.candidate))
            ]
            builds = [future.result() for future in futures]
    assert builds[0]["runner_source_sha256"] == builds[1]["runner_source_sha256"]
    assert builds[0]["resolved_lock_sha256"] == builds[1]["resolved_lock_sha256"]
    metadata["builds"] = builds
    trace_root = Path("/tmp/aisimulate-replay-parity-mooncake")
    trace_root.mkdir(exist_ok=True)
    full = trace_root / "mooncake_trace.jsonl"
    if not full.exists():
        url = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/dedfbde5daa3d5bc020adfd1d6e01c1d544f2098/FAST25-release/arxiv-trace/mooncake_trace.jsonl"
        with urllib.request.urlopen(url, timeout=60) as response:
            full.write_bytes(response.read())
    assert sha256(full) == "b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711"
    lines = full.read_bytes().splitlines(keepends=True)
    assert len(lines) == 23608
    sliced = trace_root / "mooncake_trace.rows-000000-004999.jsonl"
    sliced.write_bytes(b"".join(lines[:5000]))
    assert sha256(sliced) == "3892ae19ae480b643155f0c6b9d798591cbe2e73bec6a0fa5ae3d3bc0332fb8a"
    configs = sorted((skill / "scripts/replay-parity-runner/configs").glob("*.json"))
    metadata["trace_full_sha256"] = sha256(full)
    metadata["trace_slice_sha256"] = sha256(sliced)
    metadata["slice"] = "first5000rows,unchangedarrivalandprefixorder"
    (args.output / "campaign.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def qualify(index, config):
        row = {"name": config.stem, "original_config_sha256": sha256(config), "cpu": 160 + index, "runs": []}
        settings = json.loads(config.read_text())
        row["historical_golden_expected"] = settings.pop("expected")
        row["qualification_reason"] = (
            "Historical seeds target fb7b0d56, not this baseline; qualify baseline first, "
            "then freeze identical capacities and full canonical digest for both revisions."
        )
        authored = copy.deepcopy(settings)
        pressure_required = config.stem.startswith(("sglang", "vllm"))
        observations = []

        def observe(multiplier):
            proposal = copy.deepcopy(authored)
            proposal["expected"] = {}
            for role in proposal["spec"]["engine"].values():
                if isinstance(role, dict):
                    rank = role.get("rank", role)
                    if "num_gpu_blocks" in rank:
                        rank["num_gpu_blocks"] = max(1, round(rank["num_gpu_blocks"] * multiplier))
            attempt = len(observations)
            preflight_config = args.output / (config.stem + f"-qualification-{attempt}.json")
            preflight_config.write_text(json.dumps(proposal, indent=2) + "\n")
            destination = args.output / (config.stem + f"-qualification-{attempt}")
            with destination.with_suffix(".log").open("w") as log:
                run(
                    ["taskset", "-c", str(row["cpu"]), builds[0]["binary"], str(preflight_config), str(destination)],
                    stdout=log,
                    stderr=log,
                    timeout=60,
                )
            summary = json.loads((destination / "qualification.jsonl").read_text())
            observations.append({"capacity_multiplier": multiplier, "qualification": summary})
            return proposal, summary, destination

        settings, qualified, preflight = observe(1.0)
        if pressure_required and not 1 <= qualified["pressure_records"] <= 3:
            scales = (
                [
                    0.9999,
                    0.999,
                    0.998,
                    0.995,
                    0.99,
                    0.98,
                    0.96,
                    0.94,
                    0.9,
                    0.85,
                    0.8,
                    0.75,
                    0.7,
                    0.69,
                    0.68,
                    0.67,
                    0.66,
                    0.65,
                    0.64,
                    0.63,
                    0.62,
                    0.61,
                    0.6,
                    0.5,
                ]
                if qualified["pressure_records"] == 0
                else [
                    1.0001,
                    1.001,
                    1.002,
                    1.005,
                    1.01,
                    1.02,
                    1.04,
                    1.06,
                    1.1,
                    1.15,
                    1.2,
                    1.25,
                    1.3,
                    1.4,
                    1.5,
                    1.75,
                    2.0,
                    2.5,
                    3.0,
                ]
            )
            for multiplier in scales:
                settings, qualified, preflight = observe(multiplier)
                if 1 <= qualified["pressure_records"] <= 3:
                    break
        assert qualified["completed_requests"] == 5000 and qualified["all_requests_full_output"]
        assert qualified["requests_with_reuse"] > 0 and qualified["all_pressure_records_readmitted"]
        assert not pressure_required or 1 <= qualified["pressure_records"] <= 3, observations
        if "disaggregated" in config.stem:
            assert qualified["handoffs_complete"] == 5000
        row["capacity_qualification"] = observations
        canonical = json.loads((preflight / "canonical.jsonl").read_text())
        row["requests_waiting_for_admission"] = sum(
            request["first_admit_ms"] > request["arrival_time_ms"] for request in canonical["per_request"]
        )
        assert row["requests_waiting_for_admission"] > 0
        frozen_fields = list(row["historical_golden_expected"])
        settings["expected"] = {name: qualified[name] for name in frozen_fields}
        frozen = args.output / (config.stem + "-frozen.json")
        frozen.write_text(json.dumps(settings, indent=2) + "\n")
        row["config_sha256"] = sha256(frozen)
        row["baseline_qualification"] = qualified
        for label, build_record in zip(("baseline", "candidate"), builds, strict=True):
            for repeat in (1, 2):
                destination = args.output / (config.stem + "-" + label + "-" + str(repeat))
                with destination.with_suffix(".log").open("w") as log:
                    run(
                        ["taskset", "-c", str(row["cpu"]), build_record["binary"], str(frozen), str(destination)],
                        stdout=log,
                        stderr=log,
                    )
                qualification = json.loads((destination / "qualification.jsonl").read_text())
                row["runs"].append(
                    {
                        "revision": label,
                        "repeat": repeat,
                        "canonical_sha256": sha256(destination / "canonical.jsonl"),
                        "qualification": qualification,
                    }
                )
        assert len({record["canonical_sha256"] for record in row["runs"]}) == 1, row["name"]
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(qualify, index, config) for index, config in enumerate(configs)]
        rows = [future.result() for future in futures]
    (args.output / "parity.json").write_text(json.dumps({"rows": rows, "all_passed": True}, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "processes": 4 * len(rows), "requests_per_process": 5000, "all_passed": True}))


if __name__ == "__main__":
    main()
