"""Collect local United HCCL and a coordinated32-rank cell, then restore United."""

import json
import os
import pathlib
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[3]
OUTPUT = ROOT / "hccl-local-serving-20261022"
SERVING = "glm52-united-v0522"
CONTAINER = "aisim-910c-local-serving-20261022"
IMAGE = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"


def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def interrupted(_signum, _frame):
    raise KeyboardInterrupt("Managed campaign interrupted")


def main():
    os.umask(0o022)
    signal.signal(signal.SIGTERM, interrupted)
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / "source").mkdir()
    shutil.copyfile(REPOSITORY / "python/aisimulate/collector/sglang/ascend_hccl.py", OUTPUT / "source/hccl.py")
    shutil.copyfile(ROOT / "serving-setup.json", OUTPUT / "serving-setup.json")
    state = {"status": "preflight", "commands": []}

    def save():
        (OUTPUT / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")

    def run(command, label, timeout=1900):
        print(label, flush=True)
        with (OUTPUT / (label + ".log")).open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        state["commands"].append({"label": label, "command": command, "returncode": result.returncode})
        save()
        result.check_returncode()

    if not health():
        raise RuntimeError("United must be healthy before the managed pause")
    if subprocess.run(["docker", "inspect", CONTAINER], capture_output=True).returncode == 0:
        raise RuntimeError("Task container already exists")
    save()
    try:
        run(["npu-smi", "info"], "npu-before")
        run(["docker", "stop", "--time", "120", SERVING], "pause-united", timeout=150)
        probe = subprocess.check_output(["npu-smi", "info"], text=True)
        (OUTPUT / "npu-after-pause.log").write_text(probe)
        if probe.count("No running processes found") != 8:
            raise RuntimeError("Other NPU processes remain; do not collect on occupied devices")
        command = [
            "docker", "run", "-d", "--name", CONTAINER, "--network", "host", "--privileged", "--shm-size", "16g",
            "--mount", "type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly",
            "--mount", "type=bind,src=" + str(OUTPUT) + ",dst=/campaign",
            "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
            "-e", "HCCL_IF_BASE_PORT=28500", "-e", "HCCL_CONNECT_TIMEOUT=120", "-e", "HCCL_EXEC_TIMEOUT=300",
            "--entrypoint", "/bin/sleep",
        ]
        for path in ["/etc/ascend_install.info", "/usr/local/Ascend/firmware", "/usr/local/sbin"]:
            command += ["--mount", "type=bind,src=" + path + ",dst=" + path + ",readonly"]
        run(command + [IMAGE, "infinity"], "create-task-container")
        state["status"] = "collecting_local"
        save()
        for ranks in (2, 4, 8, 16):
            run([
                "docker", "exec", CONTAINER, "timeout", "1800", "python3", "-m", "torch.distributed.run",
                "--nproc_per_node=" + str(ranks), "--master_addr=127.0.0.1", "--master_port=29924",
                "/campaign/source/hccl.py", "--output", "/campaign/hccl-" + str(ranks) + ".json",
                "--ranks-per-node", str(ranks), "--serving-setup", "/campaign/serving-setup.json",
            ], "hccl-" + str(ranks))
            data = json.loads((OUTPUT / f"hccl-{ranks}.json").read_text())
            if sum(not row["repeatable_at_10_percent"] for row in data["cases"]) * 3 >= len(data["cases"]):
                raise RuntimeError("Systematic timing variation requires investigation")
        state["status"] = "waiting_for_peer"
        save()
        deadline = time.monotonic() + 600
        while not (OUTPUT / "peer-ready.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Peer coordination expired; restoring United")
            time.sleep(5)
        state["status"] = "collecting_dual_node"
        save()
        run([
            "docker", "exec", CONTAINER, "timeout", "1800", "python3", "-m", "torch.distributed.run",
            "--nnodes=2", "--node_rank=0", "--nproc_per_node=16",
            "--master_addr=10.120.72.32", "--master_port=29925",
            "/campaign/source/hccl.py", "--output", "/campaign/hccl-32.json",
            "--ranks-per-node", "16", "--serving-setup", "/campaign/serving-setup.json",
        ], "hccl-32")
        state["status"] = "collected"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
    finally:
        state["restore_started"] = True
        save()
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, timeout=120)
        run(["docker", "start", SERVING], "restore-united", timeout=120)
        deadline = time.monotonic() + 1800
        while not health() and time.monotonic() < deadline:
            time.sleep(10)
        state["service_healthy"] = health()
        save()
        for path in [OUTPUT, *OUTPUT.rglob("*")]:
            shutil.chown(path, user="youxingye", group="youxingye")
    if not state["service_healthy"] or state["status"] != "collected":
        raise RuntimeError(state)


if __name__ == "__main__":
    main()
