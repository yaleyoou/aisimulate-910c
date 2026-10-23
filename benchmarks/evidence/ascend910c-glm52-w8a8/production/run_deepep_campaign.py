"""Run a frozen SGLang transport campaign on authorized idle NPU nodes."""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--nodes", type=int, choices=(1, 2), default=1)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--master-address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29975)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--negative-control", choices=("dispatch", "combine"))
    parser.add_argument("--coordinator-output", type=Path)
    args = parser.parse_args()
    if socket.gethostname() not in ("npu1-19", "npu1-32") or args.node_rank >= args.nodes:
        parser.error("Use an authorized node and valid rank")
    if not re.fullmatch(r"[a-z0-9-]+", args.run_name):
        parser.error("Use a simple fresh run name")
    if args.nodes == 2 and args.node_rank == 1 and not args.coordinator_output:
        parser.error("The second node needs the shared coordinator output directory")
    os.umask(0o022)
    production = Path(__file__).resolve().parent
    repository = production.parents[3]
    output = production / args.run_name
    output.mkdir(exist_ok=False)
    coordinator = args.coordinator_output.resolve() if args.coordinator_output else output
    source = repository / "python/aisimulate/collector/sglang/ascend_deepep.py"
    shutil.copyfile(source, output / "collector.py")
    shutil.copyfile(args.plan, output / "plan.json")
    shutil.copyfile(production / "serving-setup.json", output / "setup.json")
    metadata = production / "deepep-contract-node19-20261024/model-metadata"
    before = subprocess.check_output(["npu-smi", "info"], text=True)
    (output / "npu-before.log").write_text(before)
    if before.count("No running processes found in NPU") != 8:
        raise RuntimeError("An NPU process is present")
    image = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"
    container = "aisim-" + args.run_name
    create = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "--network",
        "host",
        "--privileged",
        "--shm-size",
        "32g",
        "--mount",
        f"type=bind,src={output},dst=/campaign",
        "--mount",
        f"type=bind,src={metadata},dst=/model-metadata,readonly",
    ]
    for path in (
        "/usr/local/Ascend/driver",
        "/usr/local/Ascend/firmware",
        "/usr/local/sbin",
        "/etc/ascend_install.info",
    ):
        create.extend(("--mount", f"type=bind,src={path},dst={path},readonly"))
    create.extend(("--entrypoint", "/bin/sleep", image, "infinity"))
    command = [
        "docker",
        "exec",
        container,
        "timeout",
        "7200",
        "torchrun",
        "--nproc-per-node",
        "16",
        "--nnodes",
        str(args.nodes),
        "--node-rank",
        str(args.node_rank),
        "--master-addr",
        args.master_address,
        "--master-port",
        str(args.port),
        "/campaign/collector.py",
        "--plan",
        "/campaign/plan.json",
        "--setup",
        "/campaign/setup.json",
        "--model-metadata",
        "/model-metadata",
        "--output",
        "/campaign/observations.json",
    ]
    if args.case_index is not None:
        command.extend(("--case-index", str(args.case_index)))
    if args.negative_control:
        command.extend(("--negative-control", args.negative_control))
    state = {
        "status": "preflight",
        "host": socket.gethostname(),
        "create_command": create,
        "command": command,
        "collector_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "contamination": [],
        "attempts": [],
    }
    done = threading.Event()

    def save():
        (output / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")

    def owned_pids():
        result = subprocess.check_output(["docker", "top", container, "-eo", "pid"], text=True)
        return {int(line.strip()) for line in result.splitlines()[1:]}

    def monitor():
        known = set()
        while not done.wait(15):
            try:
                known.update(owned_pids())
                snapshot = subprocess.check_output(["npu-smi", "info"], text=True)
                active = {int(value) for value in re.findall(r"\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|", snapshot)}
                if active - known:
                    known.update(owned_pids())
                if active - known:
                    (output / "external-processes.log").write_text(snapshot)
                    raise RuntimeError(f"External NPU processes: {sorted(active - known)}")
            except Exception as error:
                state["contamination"].append(str(error))
                subprocess.run(["docker", "stop", "--time", "10", container], check=False)
                return

    save()
    subprocess.run(create, check=True)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        state["status"] = "collecting"
        save()
        plan = json.loads((output / "plan.json").read_text())
        selected = [index for index, case in enumerate(plan["cases"]) if case["num_ranks"] == args.nodes * 16]
        next_index, attempt, failures = selected[0], 0, 0
        while True:
            invocation = command.copy()
            invocation[invocation.index("/campaign/observations.json")] = f"/campaign/observations-{attempt:03}.json"
            invocation[invocation.index("--master-port") + 1] = str(args.port + attempt)
            invocation.extend(("--start-index", str(next_index)))
            with (output / f"collector-{attempt:03}.log").open("w") as stream:
                result = subprocess.run(invocation, stdout=stream, stderr=subprocess.STDOUT)
            state["attempts"].append({"attempt": attempt, "command": invocation, "returncode": result.returncode})
            save()
            if state["contamination"]:
                raise RuntimeError("Collection overlapped an external process")
            decision_path = coordinator / f"decision-{attempt:03}.json"
            if args.node_rank == 0:
                record_path = output / f"observations-{attempt:03}.json"
                record = json.loads(record_path.read_text()) if record_path.is_file() else {}
                decision = {"action": "abort", "reason": f"Collector exited {result.returncode}"}
                if args.negative_control:
                    if (
                        result.returncode != 0
                        and record.get("status") == "failed"
                        and "numerical check failed" in record.get("error", "")
                    ):
                        decision = {"action": "negative_control_rejected"}
                elif result.returncode == 0 and record.get("status") == "completed":
                    decision = {"action": "completed"}
                elif record.get("status") == "failed" and record.get("error") and args.case_index is None:
                    failed = record.get("failed_plan_index")
                    completed = [case["plan_index"] for case in record["cases"]]
                    remaining = [index for index in selected if index >= next_index]
                    if (
                        type(failed) is int
                        and completed == remaining[: len(completed)]
                        and len(completed) < len(remaining)
                        and remaining[len(completed)] == failed
                    ):
                        failures += 1
                        decision = {
                            "action": "continue",
                            "next_index": failed + 1,
                            "classification": "unexpected",
                            "failed_plan_index": failed,
                            "error": record["error"],
                            "case": plan["cases"][failed],
                        }
                        if failures >= 3 and failures / len(selected) >= 0.1:
                            decision = {"action": "abort", "reason": "Systematic transport failures require inspection"}
                        elif failed == selected[-1]:
                            decision["action"] = "completed_with_failures"
                temporary = decision_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(decision, indent=2) + "\n")
                temporary.replace(decision_path)
            else:
                started = time.monotonic()
                while not decision_path.is_file():
                    if time.monotonic() - started > 360:
                        raise RuntimeError("Missing coordinator decision")
                    time.sleep(1)
                decision = json.loads(decision_path.read_text())
            state["attempts"][-1]["decision"] = decision
            save()
            if decision["action"] == "abort":
                raise RuntimeError(decision["reason"])
            if decision["action"] != "continue":
                state["status"] = decision["action"]
                break
            next_index = decision["next_index"]
            attempt += 1
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        done.set()
        watcher.join(timeout=45)
        save()
        subprocess.run(["docker", "stop", "--time", "10", container], check=False)
        for path in (output, *output.rglob("*")):
            os.chown(path, 5031, 5031)


if __name__ == "__main__":
    main()
