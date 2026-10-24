"""Run fresh DCP attention probes with node ownership and immutable source copies."""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=("npu1-19", "npu1-32"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--world", type=int, choices=(2, 4, 8, 16), required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != args.node or not re.fullmatch(r"[a-z0-9-]+", args.run_name):
        parser.error("Use the authorized node and a simple fresh run name")
    production = Path(__file__).resolve().parent
    repository = production.parents[3]
    output = production / args.run_name
    output.mkdir(exist_ok=False)
    before = subprocess.check_output(["npu-smi", "info"], text=True)
    (output / "npu-before.log").write_text(before)
    if before.count("No running processes found in NPU") != 8:
        raise RuntimeError("NPU process inventory is not empty")
    for source, name in (
        (repository / "python/aisimulate/collector/sglang/ascend_dcp_attention.py", "collector.py"),
        (args.plan, "plan.json"),
        (production / "serving-setup.json", "serving-setup.json"),
    ):
        shutil.copyfile(source, output / name)
    container = "aisim-" + args.run_name
    image = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "--network",
        "host",
        "--privileged",
        "--shm-size",
        "16g",
        "--mount",
        f"type=bind,src={output},dst=/campaign",
        "--mount",
        f"type=bind,src={production / 'deepep-contract-node19-20261024/model-metadata'},dst=/model,readonly",
    ]
    for path in (
        "/usr/local/Ascend/driver",
        "/usr/local/Ascend/firmware",
        "/usr/local/sbin",
        "/etc/ascend_install.info",
    ):
        command.extend(("--mount", f"type=bind,src={path},dst={path},readonly"))
    command.extend(("--entrypoint", "/bin/sleep", image, "infinity"))
    state = {
        "status": "preflight",
        "node": args.node,
        "create_command": command,
        "collector_source_sha256": hashlib.sha256((output / "collector.py").read_bytes()).hexdigest(),
    }

    def save():
        (output / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    subprocess.run(command, check=True)
    done = threading.Event()
    contamination = []

    def monitor():
        known = set()
        while not done.wait(15):
            try:
                processes = subprocess.check_output(["docker", "top", container, "-eo", "pid"], text=True)
                known.update(int(line.strip()) for line in processes.splitlines()[1:])
                inventory = subprocess.check_output(["npu-smi", "info"], text=True)
                active = {int(value) for value in re.findall(r"\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|", inventory)}
                if active - known:
                    processes = subprocess.check_output(["docker", "top", container, "-eo", "pid"], text=True)
                    known.update(int(line.strip()) for line in processes.splitlines()[1:])
                if active - known:
                    contamination.append({"external_pids": sorted(active - known)})
                    (output / "external-processes.log").write_text(inventory)
                    subprocess.run(["docker", "stop", "--time", "30", container], check=False)
                    return
            except Exception as error:
                contamination.append({"monitor_error": f"{type(error).__name__}: {error}"})
                subprocess.run(["docker", "stop", "--time", "30", container], check=False)
                return

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    invocation = [
        "docker",
        "exec",
        container,
        "timeout",
        "14400",
        "torchrun",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(args.world),
        "/campaign/collector.py",
        "--plan",
        "/campaign/plan.json",
        "--serving-setup",
        "/campaign/serving-setup.json",
        "--model-metadata",
        "/model",
        "--output",
        "/campaign/observations.json",
    ]
    if args.case_index is not None:
        invocation.extend(("--case-index", str(args.case_index)))
    if args.negative_control:
        invocation.append("--negative-control")
    plan = json.loads((output / "plan.json").read_text())
    expected_indices = [
        index
        for index, case in enumerate(plan["cases"])
        if case["dcp_size"] == args.world
        and index >= args.start_index
        and (args.case_index is None or index == args.case_index)
    ]
    if args.start_index < 0 or not expected_indices:
        raise ValueError("DCP continuation must select an untouched nonempty suffix")
    state.update(
        schema="aisimulate-ascend-dcp-campaign/v1",
        status="running",
        command=invocation,
        planned_indices=expected_indices,
        attempts=[],
        failed_cases=[],
        negative_control=args.negative_control,
    )
    save()
    try:
        seen = set()
        consecutive_failures = 0
        while len(seen) < len(expected_indices):
            start = next(index for index in expected_indices if index not in seen)
            attempt = len(state["attempts"])
            name = "observations" if args.case_index is not None else f"observations-{attempt:04d}"
            current = list(invocation)
            current[current.index("--output") + 1] = f"/campaign/{name}.json"
            current.extend(("--start-index", str(start)))
            state["command"] = current
            save()
            log_name = "execution.log" if args.case_index is not None else f"execution-{attempt:04d}.log"
            with (output / log_name).open("w") as log:
                result = subprocess.run(current, stdout=log, stderr=subprocess.STDOUT)
            observed = output / (name + ".json")
            record = json.loads(observed.read_text()) if observed.is_file() else {}
            failures = sorted(output.glob(name + ".rank-*.failure.json"))
            numeric_failures = [
                json.loads(path.read_text())
                for path in failures
                if "DCP execution numerical check failed" in path.read_text()
            ]
            completed = [row["plan_index"] for row in record.get("cases", [])]
            remaining = [index for index in expected_indices if index >= start]
            if completed != remaining[: len(completed)] or set(completed) & seen:
                raise RuntimeError("DCP attempt is not the next untouched prefix of the frozen plan")
            seen.update(completed)
            entry = {
                "command": current,
                "returncode": result.returncode,
                "observed_status": record.get("status"),
                "observation": observed.name,
                "log": log_name,
                "completed_indices": completed,
                "failure_files": [path.name for path in failures],
            }
            state["attempts"].append(entry)
            state.update(returncode=result.returncode, observed_status=record.get("status"))
            if contamination:
                raise RuntimeError("DCP collection overlapped another NPU process")
            if args.negative_control:
                if result.returncode == 0 or not numeric_failures:
                    raise RuntimeError("DCP graph no-op was not rejected by numerical checks")
                seen.update(expected_indices)
                break
            if result.returncode == 0 and record.get("status") == "completed":
                if completed != remaining:
                    raise RuntimeError("DCP worker reported completion with missing cases")
                break
            if args.case_index is not None or not numeric_failures:
                raise RuntimeError("DCP probe failed; original logs and per-rank errors retained")
            failed_index = numeric_failures[0]["plan_index"]
            if failed_index != remaining[len(completed)] or any(
                failure["plan_index"] != failed_index for failure in numeric_failures
            ):
                raise RuntimeError("DCP failure does not identify the next planned case")
            seen.add(failed_index)
            state["failed_cases"].append(
                {
                    "plan_index": failed_index,
                    "classification": "numerical_check_failed",
                    "observation": observed.name,
                    "failure_files": entry["failure_files"],
                }
            )
            consecutive_failures = 1 if completed else consecutive_failures + 1
            state["attempted_cases"] = len(seen)
            save()
            if len(state["failed_cases"]) / len(seen) >= 0.1 or consecutive_failures >= 3:
                raise RuntimeError(
                    "DCP numerical failures reached the investigation threshold; remaining cases untouched"
                )
        if set(expected_indices) != seen:
            raise RuntimeError("DCP campaign is incomplete")
        state.update(status="completed", attempted_cases=len(seen))
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        done.set()
        watcher.join(timeout=45)
        state["contamination"] = contamination
        save()
        subprocess.run(["docker", "stop", "--time", "30", container], check=False)
        for path in [output, *output.rglob("*")]:
            os.chown(path, 5031, 5031)


if __name__ == "__main__":
    main()
