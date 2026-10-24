"""Run an owned single-rank Ascend transfer probe on node32 or node19."""

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
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--master-port", type=int, default=30200)
    parser.add_argument("--store-port", type=int, default=30107)
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--source-tp", type=int)
    parser.add_argument("--destination-tp", type=int)
    parser.add_argument("--case-index", type=int)
    args = parser.parse_args()
    node = ("npu1-32", "npu1-19")[args.node_rank]
    address = ("10.120.72.32", "10.120.72.19")[args.node_rank]
    if socket.gethostname() != node or not re.fullmatch(r"[a-z0-9-]+", args.suffix):
        parser.error("Use the requested node and a simple fresh suffix")
    os.umask(0o022)
    production = Path(__file__).resolve().parent
    output = production / ("pd-probe-" + node + "-" + args.suffix)
    output.mkdir(exist_ok=False)
    source = (
        production.parents[3] / "python/aisimulate/collector/sglang/ascend_pd.py"
        if args.plan
        else production / "probe_pd_transfer.py"
    )
    shutil.copyfile(source, output / source.name)
    if args.plan:
        if args.source_tp is None or args.destination_tp is None:
            parser.error("A planned campaign needs both attention TP sizes")
        shutil.copyfile(args.plan, output / "plan.json")
    shutil.copyfile(production / "serving-setup.json", output / "setup.json")
    before = subprocess.check_output(["npu-smi", "info"], text=True)
    (output / "npu-before.log").write_text(before)
    if before.count("No running processes found in NPU") != 8:
        raise RuntimeError("Another NPU process is present")
    metadata = production / "deepep-contract-node19-20261024/model-metadata"
    container = "aisim-pd-probe-" + args.suffix
    image = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"
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
        "16g",
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
        "3600" if args.plan else "600",
        "torchrun",
        "--nnodes",
        "2",
        "--nproc-per-node",
        "16" if args.plan else "1",
        "--node-rank",
        str(args.node_rank),
        "--master-addr",
        "10.120.72.32",
        "--master-port",
        str(args.master_port),
        "/campaign/" + source.name,
        "--local-ip",
        address,
        "--store-url",
        "tcp://10.120.72.32:" + str(args.store_port),
        "--setup",
        "/campaign/setup.json",
        "--model-metadata",
        "/model-metadata",
        "--output",
        "/campaign/observations.json",
    ]
    if args.plan:
        command.extend(
            (
                "--plan",
                "/campaign/plan.json",
                "--source-tp",
                str(args.source_tp),
                "--destination-tp",
                str(args.destination_tp),
            )
        )
        if args.case_index is not None:
            command.extend(("--case-index", str(args.case_index)))
    else:
        command.extend(("--tokens", str(args.tokens)))
    if args.negative_control:
        command.append("--negative-control")
    state = {
        "status": "preflight",
        "host": node,
        "create_command": create,
        "command": command,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "contamination": [],
    }
    done = threading.Event()

    def save():
        (output / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")

    def monitor():
        known = set()
        while not done.wait(15):
            try:
                processes = subprocess.check_output(["docker", "top", container, "-eo", "pid"], text=True)
                known.update(int(line.strip()) for line in processes.splitlines()[1:])
                snapshot = subprocess.check_output(["npu-smi", "info"], text=True)
                active = {int(value) for value in re.findall(r"\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|", snapshot)}
                if active - known:
                    processes = subprocess.check_output(["docker", "top", container, "-eo", "pid"], text=True)
                    known.update(int(line.strip()) for line in processes.splitlines()[1:])
                if active - known:
                    (output / "external-processes.log").write_text(snapshot)
                    raise RuntimeError("Another NPU owner appeared")
            except Exception as error:
                state["contamination"].append(str(error))
                subprocess.run(["docker", "stop", "--time", "10", container], check=False)
                return

    save()
    subprocess.run(create, check=True)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        state["status"] = "running"
        save()
        with (output / "probe.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        state["returncode"] = result.returncode
        path = output / "observations.json"
        record = json.loads(path.read_text()) if path.is_file() else {}
        if state["contamination"]:
            raise RuntimeError("Probe overlapped another NPU owner")
        if args.negative_control:
            peer_node = "32" if args.plan else "19"
            peer_path = production / ("pd-probe-npu1-" + peer_node + "-" + args.suffix) / "observations.json"
            receiver = json.loads(peer_path.read_text()) if peer_path.is_file() else {}
            expected = (
                result.returncode != 0
                and receiver.get("status") == "failed"
                and ("numerical check failed" in receiver.get("error", ""))
            )
            if not expected:
                raise RuntimeError("No-transfer control did not fail")
            state["status"] = "negative_control_rejected"
        else:
            expected_status = "completed" if args.plan else "verified"
            if (
                result.returncode != 0
                or (not args.plan or args.node_rank == 0)
                and record.get("status") != expected_status
            ):
                raise RuntimeError(f"Probe exited {result.returncode}, status={record.get('status')}")
            state["status"] = "completed"
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
