import datetime
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import time
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent
CONTAINER = "aisim-910c-hardware-20261021"
SERVICE = "glm52-united-v0522"
SSH = ["ssh", "-F", "/dev/null", "-p", "22290", "-i", "/root/.ssh/aisim_key",
       "-oBatchMode=yes", "-oStrictHostKeyChecking=accept-new", "root@10.120.72.19"]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(command, output, label, timeout=240):
    started = now()
    print(started, label, flush=True)
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (output / (label + ".log")).write_text(result.stdout + result.stderr)
    with (output / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"started": started, "finished": now(), "command": command,
                                 "returncode": result.returncode}) + "\n")
    result.check_returncode()
    return result.stdout


def execute(command, output, label, timeout=240):
    return run(["docker", "exec", CONTAINER] + command, output, label, timeout)


def mpi(ranks, per_node, dual):
    command = ["timeout", "180", "mpirun", "--allow-run-as-root", "--bind-to", "none",
               "--mca", "btl_tcp_if_include", "bond4", "--mca", "oob_tcp_if_include", "bond4"]
    if dual:
        command += ["--mca", "plm_rsh_args", "-F /dev/null -p 22290 -i /root/.ssh/aisim_key -oBatchMode=yes -oStrictHostKeyChecking=accept-new",
                    "--host", f"10.120.72.32:{per_node},10.120.72.19:{per_node}"]
    command += ["-np", str(ranks)]
    for variable in ["PATH", "LD_LIBRARY_PATH", "HCCL_SOCKET_IFNAME", "HCCL_IF_BASE_PORT", "HCCL_BUFFSIZE"]:
        command += ["-x", variable]
    return command


def check_idle(text):
    processes = text.split("Process id", 1)[-1]
    if re.search(r"^\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|", processes, re.MULTILINE):
        raise RuntimeError("NPU processes remain; exclusive collection refused")


def healthy():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    output = ROOT / ("run-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    output.mkdir()
    (ROOT / "current-run.txt").write_text(str(output))
    source_dir = output / "source"
    source_dir.mkdir()
    for source in ROOT.glob("*.py"):
        shutil.copy2(source, source_dir / source.name)
    shutil.copy2(ROOT / "p2p_probe.cpp", source_dir / "p2p_probe.cpp")
    manifest = {"started": now(), "status": "preflight", "service_restored": False,
                "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_dir.iterdir()},
                "hosts": ["npu1-32", "npu1-19"], "scope": "same-superpod hardware calibration",
                "cv_limit": 0.10, "repeats": 9, "iterations": 50, "warmup": 50}

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    stopped = False
    try:
        if not healthy():
            raise RuntimeError("Serving service is not healthy before collection")
        peer = execute(SSH + ["npu-smi", "info"], output, "remote-before")
        check_idle(peer)
        run(["docker", "stop", "--time", "60", SERVICE], output, "stop-service", timeout=90)
        stopped = True
        check_idle(run(["npu-smi", "info"], output, "local-exclusive"))
        manifest["status"] = "collecting"
        save()
        cases = [
            ("sio-2", 2, 2, False, "0,1"),
            ("hccs-2", 2, 2, False, "0,2"),
            ("local-16", 16, 16, False, ",".join(map(str, range(16)))),
            ("cross-2", 2, 1, True, "0"),
            ("cross-32", 32, 16, True, ",".join(map(str, range(16)))),
        ]
        for label, ranks, per_node, dual, devices in cases:
            prefix = "/tmp/aisim-" + label
            execute(mpi(ranks, per_node, dual) + ["/tmp/p2p_probe", devices, prefix, "50", "9", "268435456"], output, label)
            for rank in range(ranks):
                command = ["cat", prefix + "-rank" + str(rank) + ".jsonl"]
                if dual and rank >= per_node:
                    command = SSH + command
                text = execute(command, output, label + "-fetch-" + str(rank))
                (output / (label + "-rank" + str(rank) + ".jsonl")).write_text(text)
        for dual in [False, True]:
            ranks = 32 if dual else 16
            topology = "cross-32" if dual else "local-16"
            for operation in ["all_reduce", "all_gather", "reduce_scatter", "alltoall"]:
                for repeat in range(3 if operation == "all_reduce" else 1):
                    label = topology + "-" + operation + "-" + str(repeat)
                    binary = "/usr/local/Ascend/cann/tools/hccl_test/bin/" + operation + "_test"
                    execute(mpi(ranks, 16, dual) + [binary, "-b", "1K", "-e", "256M", "-f", "8",
                            "-d", "bfp16", "-p", "16", "-n", "50", "-w", "50", "-t", "1", "-c", "1"],
                            output, label)
        manifest["status"] = "collected"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["collection_finished"] = now()
        save()
        if stopped:
            run(["docker", "start", SERVICE], output, "restore-service", timeout=60)
            manifest["service_restart_requested"] = now()
            save()
            deadline = time.monotonic() + 2400
            while time.monotonic() < deadline:
                if healthy():
                    manifest["service_restored"] = True
                    manifest["service_ready"] = now()
                    break
                time.sleep(10)
            save()
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
