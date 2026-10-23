"""Cover the global dense-MLP payloads observed with both attention-DP ranks active."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess


def main():
    if socket.gethostname() != "npu1-32":
        raise RuntimeError("This campaign is assigned to the idle node32")
    os.umask(0o022)
    production = Path(__file__).resolve().parent
    repository = production.parents[3]
    output = production / "hccl-large-local-node32-20261023"
    output.mkdir(exist_ok=False)
    shutil.copyfile(repository / "python/aisimulate/collector/sglang/ascend_hccl.py", output / "collector.py")
    shutil.copyfile(production / "serving-setup.json", output / "serving-setup.json")
    before = subprocess.check_output(["npu-smi", "info"], text=True)
    (output / "npu-before.log").write_text(before)
    if before.count("No running processes found in NPU") != 8:
        raise RuntimeError("Another NPU process is present")
    container = "aisim-hccl-large-20261023"
    image = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"
    command = ["docker", "run", "-d", "--name", container, "--network", "host", "--privileged", "--shm-size", "16g", "--mount", f"type=bind,src={output},dst=/campaign", "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "HCCL_IF_BASE_PORT=28800", "-e", "HCCL_CONNECT_TIMEOUT=120", "-e", "HCCL_EXEC_TIMEOUT=300"]
    for path in ("/usr/local/Ascend/driver", "/usr/local/Ascend/firmware", "/usr/local/sbin", "/etc/ascend_install.info"):
        command.extend(("--mount", f"type=bind,src={path},dst={path},readonly"))
    command.extend(("--entrypoint", "/bin/sleep", image, "infinity"))
    state = {"status": "preflight", "create_command": command, "collector_source_sha256": hashlib.sha256((output / "collector.py").read_bytes()).hexdigest(), "driver_version_info": Path("/usr/local/Ascend/driver/version.info").read_text(), "jobs": [], "scope": "192/384MiB input payloads; actual dense global-token observations require192MiB at M16384,H6144,BF16"}

    def save():
        (output / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    subprocess.run(command, check=True)
    try:
        for ranks in (2, 4, 8, 16):
            state.update(status="collecting", ranks=ranks)
            save()
            invocation = ["docker", "exec", container, "timeout", "2400", "python3", "-m", "torch.distributed.run", f"--nproc_per_node={ranks}", "--master_addr=127.0.0.1", "--master_port=29961", "/campaign/collector.py", "--output", f"/campaign/hccl-{ranks}.json", "--ranks-per-node", str(ranks), "--serving-setup", "/campaign/serving-setup.json", "--sizes", "201326592,402653184"]
            with (output / f"hccl-{ranks}.log").open("w") as stream:
                result = subprocess.run(invocation, stdout=stream, stderr=subprocess.STDOUT, timeout=2500)
            state["jobs"].append({"ranks": ranks, "command": invocation, "returncode": result.returncode})
            save()
            result.check_returncode()
            record = json.loads((output / f"hccl-{ranks}.json").read_text())
            if record["status"] != "completed" or len(record["cases"]) != 16:
                raise ValueError("Incomplete collective grid")
        state["status"] = "completed"
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        subprocess.run(["docker", "stop", "--time", "30", container], check=False)
        for path in (output, *output.rglob("*")):
            os.chown(path, 5031, 5031)


if __name__ == "__main__":
    main()
