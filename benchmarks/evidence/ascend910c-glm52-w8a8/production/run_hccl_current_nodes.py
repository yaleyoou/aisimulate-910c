"""Collect current 31/32 HCCL groups while preserving existing stopped services."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    os.umask(0o022)
    expected_host = "npu1-32" if args.node_rank == 0 else "npu1-31"
    if socket.gethostname() != expected_host:
        raise ValueError("Unexpected collection host")
    shared = Path(__file__).resolve().parent / "hccl-node31-node32-20261023"
    output = shared / expected_host
    output.mkdir(exist_ok=False)
    source = shared / "source/hccl.py"
    setup = shared / "serving-setup.json"
    if not source.is_file() or not setup.is_file():
        raise ValueError("Freeze source and serving setup before launching workers")
    image = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"
    before = subprocess.check_output(["npu-smi", "info"], text=True)
    (output / "npu-before.log").write_text(before)
    if before.count("No running processes found in NPU") != 8:
        raise RuntimeError("Another NPU process is present")
    container = "aisim-hccl-current-20261023"
    command = ["docker", "run", "-d", "--name", container, "--network", "host", "--privileged", "--shm-size", "16g", "--mount", f"type=bind,src={shared},dst=/campaign", "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "HCCL_IF_BASE_PORT=28700", "-e", "HCCL_CONNECT_TIMEOUT=120", "-e", "HCCL_EXEC_TIMEOUT=300"]
    for path in ("/usr/local/Ascend/driver", "/usr/local/Ascend/firmware", "/usr/local/sbin", "/etc/ascend_install.info"):
        command.extend(("--mount", f"type=bind,src={path},dst={path},readonly"))
    command.extend(("--entrypoint", "/bin/sleep", image, "infinity"))
    state = {"status": "preflight", "host": expected_host, "node_rank": args.node_rank, "image_digest": image, "create_command": command, "collector_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "serving_setup_sha256": hashlib.sha256(setup.read_bytes()).hexdigest(), "driver_version_info": Path("/usr/local/Ascend/driver/version.info").read_text(), "jobs": []}

    def save():
        (output / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    subprocess.run(command, check=True)
    try:
        state["status"] = "ready"
        save()
        (output / "ready.json").write_text(json.dumps({"host": expected_host, "source_sha256": state["collector_source_sha256"], "setup_sha256": state["serving_setup_sha256"]}) + "\n")
        peer = shared / ("npu1-31" if args.node_rank == 0 else "npu1-32") / "ready.json"
        deadline = time.monotonic() + 300
        while not peer.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError("Peer readiness expired")
            time.sleep(1)
        peer_record = json.loads(peer.read_text())
        if peer_record["source_sha256"] != state["collector_source_sha256"] or peer_record["setup_sha256"] != state["serving_setup_sha256"]:
            raise ValueError("Peer source/setup identity mismatch")
        for local_ranks in (8, 16):
            state.update(status="collecting", ranks_per_node=local_ranks)
            save()
            invocation = ["docker", "exec", container, "timeout", "1800", "python3", "-m", "torch.distributed.run", "--nnodes=2", f"--node_rank={args.node_rank}", f"--nproc_per_node={local_ranks}", "--master_addr=10.120.72.32", f"--master_port={29940+local_ranks}", "/campaign/source/hccl.py", "--output", f"/campaign/hccl-{2*local_ranks}.json", "--ranks-per-node", str(local_ranks), "--serving-setup", "/campaign/serving-setup.json"]
            with (output / f"hccl-{2*local_ranks}.log").open("w") as stream:
                result = subprocess.run(invocation, stdout=stream, stderr=subprocess.STDOUT, timeout=1900)
            state["jobs"].append({"num_ranks": 2*local_ranks, "ranks_per_node": local_ranks, "command": invocation, "returncode": result.returncode})
            save()
            result.check_returncode()
        state["status"] = "completed"
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        subprocess.run(["docker", "stop", "--time", "30", container], check=False)
        for path in (output, *output.rglob("*")):
            os.chown(path,5031,5031)
        if args.node_rank == 0:
            for path in shared.glob("hccl-*.json"):
                os.chown(path,5031,5031)


if __name__ == "__main__":
    main()
