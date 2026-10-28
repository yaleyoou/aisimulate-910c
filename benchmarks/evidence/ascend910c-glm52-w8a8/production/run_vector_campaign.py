"""Verify every vector branch, reject no-op replay, then collect the complete grid."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="vector-full-node32-20261023")
    parser.add_argument("--operation", choices=("vector", "moe_chain", "moe_tensor", "bmm", "linear"), default="vector")
    parser.add_argument("--node", default=socket.gethostname())
    parser.add_argument("--existing-plan", type=Path)
    args = parser.parse_args()
    if socket.gethostname() != args.node or not re.fullmatch(r"[a-z0-9-]+", args.run_name):
        parser.error("Use the requested node and a simple fresh run name")
    os.umask(0o022)
    os.sched_setaffinity(0, [300, 620])
    production = Path(__file__).resolve().parent
    repository = production.parents[3]
    output = production / args.run_name
    output.mkdir(exist_ok=False)
    image = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"
    planner = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONPATH=/workspace/python/aisimulate",
        "--mount",
        f"type=bind,src={repository},dst=/workspace,readonly",
        "--mount",
        "type=bind,src=/mnt/raid/user/data/models/glm-q,dst=/model,readonly",
        "--entrypoint",
        "python3",
        image,
        "-c",
        "import json;from collector.sglang.ascend_vector_cases import get_ascend_vector_cases;plan=get_ascend_vector_cases('/model');plan['model_path']='/mnt/raid/user/data/models/glm-q';print(json.dumps(plan,indent=2))",
    ]
    planner[-1] = planner[-1].replace("ascend_vector", "ascend_" + args.operation)
    if args.existing_plan:
        planner = ["frozen-plan", str(args.existing_plan.resolve())]
        plan = json.loads(args.existing_plan.read_text())
    else:
        plan = json.loads(subprocess.check_output(planner, text=True))
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    source_names = ["ascend_" + args.operation + ".py"]
    if args.operation == "vector":
        source_names.append("ascend_vector_kernels.py")
    for name in source_names:
        shutil.copyfile(repository / "python/aisimulate/collector/sglang" / name, output / name)
    shutil.copyfile(production / "serving-setup.json", output / "serving-setup-nodes.json")
    shutil.copyfile(production / "hccl_graph_noop_probe.py", output / "graph_noop.py")
    before = subprocess.check_output(["npu-smi", "info"], text=True)
    (output / "npu-before.log").write_text(before)
    if before.count("No running processes found in NPU") != 8:
        raise RuntimeError("Another NPU process is present")
    container = "aisim-" + args.run_name
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
        "operation": args.operation,
        "node": args.node,
        "planner_command": planner,
        "create_command": command,
        "jobs": [],
        "source_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in source_names},
    }
    done = threading.Event()
    contamination = []

    def save():
        (output / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")

    def owned_pids():
        text = subprocess.check_output(["docker", "top", container, "-eo", "pid"], text=True)
        return {int(line.strip()) for line in text.splitlines()[1:]}

    def monitor():
        known = set()
        while not done.wait(15):
            try:
                known.update(owned_pids())
                text = subprocess.check_output(["npu-smi", "info"], text=True)
                active = {int(value) for value in re.findall(r"\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|", text)}
                if active - known:
                    known.update(owned_pids())
                if active - known:
                    contamination.append({"external_pids": sorted(active - known)})
                    (output / "external-processes.log").write_text(text)
                    subprocess.run(["docker", "stop", "--time", "30", container], check=False)
                    return
            except Exception as error:
                contamination.append({"monitor_error": f"{type(error).__name__}: {error}"})
                subprocess.run(["docker", "stop", "--time", "30", container], check=False)
                return

    def run(name, device, extra, negative=False):
        arguments = ["python3"]
        if negative:
            arguments.append("/campaign/graph_noop.py")
        arguments.extend(
            (
                "/campaign/ascend_" + args.operation + ".py",
                "--plan",
                "/campaign/plan.json",
                "--serving-setup",
                "/campaign/serving-setup-nodes.json",
                "--device",
                str(device),
                "--output",
                f"/campaign/{name}.json",
                *extra,
            )
        )
        invocation = ["docker", "exec", container, "timeout", "7200", *arguments]
        with (output / (name + ".log")).open("w") as stream:
            result = subprocess.run(invocation, stdout=stream, stderr=subprocess.STDOUT)
        path = output / (name + ".json")
        record = json.loads(path.read_text()) if path.is_file() else {}
        expected = (
            (
                result.returncode != 0
                and record.get("status") == "failed"
                and "numerical check failed" in record.get("error", "")
            )
            if negative
            else result.returncode == 0 and record.get("status") == "completed"
        )
        return {
            "name": name,
            "command": invocation,
            "returncode": result.returncode,
            "expected_outcome": expected,
            "cases": len(record.get("cases", [])),
        }

    def parallel(prefix, extra):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(
                    run, f"{prefix}-{index}", index * 2, ["--num-shards", "4", "--shard-index", str(index), *extra]
                )
                for index in range(4)
            ]
            jobs = [future.result() for future in concurrent.futures.as_completed(futures)]
        state["jobs"].extend(jobs)
        save()
        if contamination or not all(job["expected_outcome"] for job in jobs):
            raise RuntimeError(f"{prefix} failed or overlapped another NPU process")

    save()
    subprocess.run(command, check=True)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        state["status"] = "smoke"
        save()
        smoke_filter = {
            "vector": '"tokens": 8}',
            "moe_chain": '"received_tokens": 8}',
            "moe_tensor": '"tokens": 8,',
        }.get(args.operation)
        if smoke_filter is not None and any(
            case.get("tokens", case.get("received_tokens")) == 8 for case in plan["cases"]
        ):
            parallel("smoke", ["--case-filter", smoke_filter])
        state["status"] = "negative_controls"
        save()
        controls = {}
        for index, case in enumerate(plan["cases"]):
            if case["execution_mode"] != "graph":
                continue
            if args.operation == "vector":
                controls.setdefault(case["role"], index)
            elif args.operation == "moe_chain" and case["received_tokens"] == 8:
                controls[case["phase"] + "-" + str(case["local_experts"]) + "-" + "-".join(case["distributions"])] = (
                    index
                )
            elif args.operation == "moe_tensor" and case["tokens"] == 8:
                controls[str(case["intermediate_size"])] = index
            elif args.operation == "bmm":
                controls.setdefault(case["role"], index)
            elif args.operation == "linear":
                controls.setdefault(case["quant_mode"] + "-" + str(case["quant_bias"]), index)
        for label, index in sorted(controls.items()):
            job = run(
                "negative-" + label, 0, ["--num-shards", str(len(plan["cases"])), "--shard-index", str(index)], True
            )
            state["jobs"].append(job)
            save()
            if contamination or not job["expected_outcome"]:
                raise RuntimeError("Graph no-op control was not rejected by numerical checks")
        state["status"] = "collecting"
        save()
        parallel("shard", [])
        state["status"] = "completed"
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", contamination=contamination)
        raise
    finally:
        done.set()
        watcher.join(timeout=45)
        save()
        subprocess.run(["docker", "stop", "--time", "30", container], check=False)
        for path in (output, *output.rglob("*")):
            os.chown(path, 5031, 5031)


if __name__ == "__main__":
    main()
