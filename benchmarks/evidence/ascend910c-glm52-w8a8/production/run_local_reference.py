"""Collect source-matched local references, then restore the existing United service."""

import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess
import time
import urllib.error
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[3]
SERVING = "glm52-united-v0522"
CONTAINER = "aisim-910c-local-reference-20261022"
IMAGE = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"


def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def main():
    output = ROOT / "local-reference-20261022"
    output.mkdir(exist_ok=False)
    sources = output / "source"
    sources.mkdir()
    files = {
        "hccl.py": ROOT / "hccl-peer-20261022/ascend_hccl.py",
        "quant.py": REPOSITORY / "python/aisimulate/collector/sglang/ascend_quant_probe.py",
    }
    state = {"status": "preflight", "commands": [], "source_sha256": {}}
    for name, path in files.items():
        shutil.copyfile(path, sources / name)
        state["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def save():
        (output / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")

    def run(command, label, timeout=2400):
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), label, flush=True)
        with (output / (label + ".log")).open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        state["commands"].append({"label": label, "command": command, "returncode": result.returncode})
        save()
        result.check_returncode()

    save()
    if not health():
        raise RuntimeError("United must be healthy before the managed pause")
    if subprocess.run(["docker", "inspect", CONTAINER], capture_output=True).returncode == 0:
        raise RuntimeError("Task container already exists")
    run(["npu-smi", "info"], "npu-before")
    try:
        run(["docker", "stop", "--time", "120", SERVING], "pause-united", timeout=150)
        command = [
            "docker", "run", "-d", "--name", CONTAINER, "--network", "host", "--privileged", "--shm-size", "16g",
            "--mount", "type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly",
            "--mount", "type=bind,src=" + str(output) + ",dst=/campaign",
            "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "TASK_QUEUE_ENABLE=1",
            "-e", "HCCL_SOCKET_IFNAME=bond4", "-e", "GLOO_SOCKET_IFNAME=bond4",
            "-e", "HCCL_IF_BASE_PORT=28500", "-e", "HCCL_CONNECT_TIMEOUT=120", "-e", "HCCL_EXEC_TIMEOUT=300",
            "--entrypoint", "/bin/sleep",
        ]
        for path in ["/etc/ascend_install.info", "/usr/local/Ascend/firmware", "/usr/local/sbin"]:
            if pathlib.Path(path).exists():
                command += ["--mount", "type=bind,src=" + path + ",dst=" + path + ",readonly"]
        run(command + [IMAGE, "infinity"], "create-task-container")
        state["status"] = "collecting"
        save()
        for ranks in [2, 4, 8, 16]:
            run([
                "docker", "exec", CONTAINER, "timeout", "1800", "python3", "-m", "torch.distributed.run",
                "--nproc_per_node=" + str(ranks), "--master_addr=127.0.0.1", "--master_port=29922",
                "/campaign/source/hccl.py", "--output", "/campaign/hccl-" + str(ranks) + ".json",
                "--ranks-per-node", str(ranks),
            ], "hccl-" + str(ranks), timeout=1900)
        for family in ["quant", "attention"]:
            run(["docker", "exec", CONTAINER, "python3", "/campaign/source/quant.py", "--family", family,
                 "--output", "/campaign/" + family + ".json"], family)
        state["status"] = "collected"
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
    finally:
        state["restore_started"] = True
        save()
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, timeout=120)
        run(["docker", "start", SERVING], "restore-united", timeout=120)
        deadline = time.monotonic() + 1500
        while not health() and time.monotonic() < deadline:
            time.sleep(10)
        state["service_healthy"] = health()
        save()
        for path in [output, *output.rglob("*")]:
            shutil.chown(path, user="youxingye", group="youxingye")
    if not state["service_healthy"] or state["status"] != "collected":
        raise RuntimeError(state)


if __name__ == "__main__":
    main()
