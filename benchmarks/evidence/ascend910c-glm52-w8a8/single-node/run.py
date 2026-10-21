import argparse
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request


IMAGE = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"
GROUPS = {"sio-2": [0, 1], "hccs-2": [0, 2], "attention-8a": list(range(8)), "attention-8b": list(range(8, 16)), "node-16": list(range(16))}


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(arguments, output, label, timeout=1200):
    print(now(), label, flush=True)
    with (output / (label + ".log")).open("w") as stream:
        result = subprocess.run(arguments, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    with (output / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"at": now(), "arguments": arguments, "exit_code": result.returncode}) + "\n")
    result.check_returncode()


def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def container_command(output, name, mode, devices, label, args):
    arguments = [
        "docker", "run", "--rm", "--name", name, "--network", "host", "--privileged", "--shm-size", "16g",
        "--mount", "type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly",
        "--mount", "type=bind,src=" + str(output / "source") + ",dst=/campaign,readonly",
        "--mount", "type=bind,src=" + str(output) + ",dst=/results",
        "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "HCCL_SOCKET_IFNAME=bond4",
        "-e", "GLOO_SOCKET_IFNAME=bond4", "-e", "HCCL_IF_BASE_PORT=26000", "-e", "HCCL_BUFFSIZE=650",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "TASK_QUEUE_ENABLE=0",
    ]
    for filename in ["/etc/ascend_install.info", "/usr/local/Ascend/firmware", "/usr/local/sbin"]:
        if pathlib.Path(filename).exists():
            arguments += ["--mount", "type=bind,src=" + filename + ",dst=" + filename + ",readonly"]
    for device in ["/dev/davinci" + str(index) for index in range(16)] + ["/dev/davinci_manager", "/dev/hisi_hdc", "/dev/devmm_svm"]:
        if pathlib.Path(device).exists():
            arguments += ["--device", device]
    arguments += ["--entrypoint", "python3", IMAGE]
    if mode == "communication":
        arguments += ["-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=" + str(len(devices)), "--master_addr=127.0.0.1", "--master_port=29460"]
    launcher = "affinity.py" if args.rank_affinity and mode == "communication" else "collect.py"
    arguments += ["/campaign/" + launcher, mode, "--devices", ",".join(map(str, devices)), "--output", "/results/" + label + ".json"]
    arguments += ["--iterations", str(args.iterations), "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
    return arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--service", default="glm52-united-v0522")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reuse-local-source", type=pathlib.Path)
    parser.add_argument("--allow-starting-service", action="store_true")
    parser.add_argument("--rank-affinity", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.warmup, args.iterations) <= 0 or args.repeats < 2:
        parser.error("Warmup/iterations must be positive and repeats must be at least two")
    output = args.output.resolve()
    plan = {
        "image": IMAGE,
        "service": args.service,
        "scope": "single-node hardware calibration, not formal FPM collection",
        "devices": list(range(16)),
        "communication_groups": GROUPS,
        "expected_single_device_points": 81,
        "expected_communication_points": 100,
        "warmup": args.warmup, "iterations_per_repeat": args.iterations, "repeats": args.repeats,
        "rank_affinity": args.rank_affinity,
        "reused_local_source": str(args.reuse_local_source.resolve()) if args.reuse_local_source else None,
        "measurement": "eager; NPU events and wall time for local cases, max-rank synchronized wall time for HCCL",
        "quality_policy": "retain every sample; flag CV above 0.10; do not discard or qualify unstable points",
        "recovery": "restart the specified initially-running service in finally and verify /health",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    inspected = json.loads(subprocess.check_output(["docker", "inspect", args.service]))[0]
    initial_health = health()
    if not inspected["State"]["Running"] or (not initial_health and not args.allow_starting_service):
        raise RuntimeError("Expected healthy running baseline service before maintenance")
    if inspected["Config"]["Image"] != IMAGE:
        raise RuntimeError("Serving image does not match the pinned calibration image")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    plan["sources_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in pathlib.Path(__file__).resolve().parent.glob("*.py")
    }
    for name in plan["sources_sha256"]:
        shutil.copy2(pathlib.Path(__file__).resolve().parent / name, output / "source" / name)
    if args.reuse_local_source:
        local_source = args.reuse_local_source.resolve()
        source_plan = json.loads((local_source.parent / "plan.json").read_text())
        if source_plan["image"] != IMAGE:
            raise ValueError("Local reference uses a different runtime image")
        local = json.loads(local_source.read_text())
        if not local["complete"] or local["errors"] or len(local["cases"]) != 81:
            raise ValueError("Local reference is incomplete")
        plan["reused_local_source_sha256"] = hashlib.sha256(local_source.read_bytes()).hexdigest()
        shutil.copy2(local_source, output / "single.json")
    save(output / "plan.json", plan)
    manifest = {"started_at": now(), "status": "preflight", "service_id": inspected["Id"], "image_id": inspected["Image"], "service_restored": False, "service_initially_healthy": initial_health}
    save(output / "manifest.json", manifest)
    command(["npu-smi", "info"], output, "npu-before", timeout=30)
    command(["npu-smi", "info", "-t", "topo"], output, "topology", timeout=30)
    stopped = False
    active_container = "aisim-910c-calibration-" + str(int(time.time()))
    try:
        stopped = True
        command(["docker", "stop", "--time", "45", args.service], output, "stop-service", timeout=55)
        manifest["status"] = "collecting"
        manifest["service_stopped_at"] = now()
        save(output / "manifest.json", manifest)
        command(["npu-smi", "info"], output, "npu-idle", timeout=30)
        processes = (output / "npu-idle.log").read_text().split("Process id", 1)[-1]
        if re.search(r"^\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|", processes, re.MULTILINE):
            raise RuntimeError("Other NPU processes remain; calibration requires exclusive devices")
        if not args.reuse_local_source:
            command(container_command(output, active_container, "single", list(range(16)), "single", args), output, "single")
        for label, devices in GROUPS.items():
            command(container_command(output, active_container, "communication", devices, label, args), output, label)
        manifest["status"] = "collected"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        subprocess.run(["docker", "stop", "--time", "10", active_container], capture_output=True, timeout=20)
        raise
    finally:
        manifest["collection_finished_at"] = now()
        save(output / "manifest.json", manifest)
        if stopped:
            command(["docker", "start", args.service], output, "restore-service", timeout=60)
            manifest["service_restart_requested_at"] = now()
            save(output / "manifest.json", manifest)
            deadline = time.monotonic() + 2400
            while time.monotonic() < deadline:
                if health():
                    manifest["service_restored"] = True
                    manifest["service_ready_at"] = now()
                    break
                time.sleep(10)
            save(output / "manifest.json", manifest)
            if not manifest["service_restored"]:
                raise RuntimeError("Service restart requested but health did not recover within 40 minutes")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
