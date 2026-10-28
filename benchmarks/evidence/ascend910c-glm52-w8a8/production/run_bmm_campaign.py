"""Run the source-matched MLA BMM grid after three graph-noop controls."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="bmm-full-node32-20261023")
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        parser.error("run-name must be a single directory name")
    os.umask(0o022)
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
        "import json;from collector.sglang.ascend_bmm_cases import get_ascend_bmm_cases;plan=get_ascend_bmm_cases('/model');plan['model_path']='/mnt/raid/user/data/models/glm-q';print(json.dumps(plan,indent=2))",
    ]
    if args.plan:
        planner = ["frozen-plan", str(args.plan.resolve())]
        plan = json.loads(args.plan.read_text())
    else:
        plan = json.loads(subprocess.check_output(planner, text=True))
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for source, name in (
        (repository / "python/aisimulate/collector/sglang/ascend_bmm.py", "collector.py"),
        (production / "serving-setup.json", "serving-setup-nodes.json"),
        (production / "hccl_graph_noop_probe.py", "graph_noop.py"),
    ):
        shutil.copyfile(source, output / name)
    before = subprocess.check_output(["npu-smi", "info"], text=True)
    (output / "npu-before.log").write_text(before)
    if before.count("No running processes found in NPU") != 8:
        raise RuntimeError("NPU process inventory is not empty")
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
    subprocess.run(command, check=True)
    state = {
        "status": "running",
        "planner_command": planner,
        "create_command": command,
        "jobs": [],
        "collector_source_sha256": hashlib.sha256((output / "collector.py").read_bytes()).hexdigest(),
    }

    def save():
        (output / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")

    def run(name, device, extra, negative=False):
        arguments = ["python3"]
        if negative:
            arguments.append("/campaign/graph_noop.py")
        arguments.extend(
            (
                "/campaign/collector.py",
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

    save()
    try:
        for role in sorted({case["role"] for case in plan["cases"]}):
            index = next(
                index
                for index, case in enumerate(plan["cases"])
                if case["role"] == role and case["execution_mode"] == "graph"
            )
            job = run(
                "negative-" + role, 0, ["--num-shards", str(len(plan["cases"])), "--shard-index", str(index)], True
            )
            state["jobs"].append(job)
            save()
            if not job["expected_outcome"]:
                raise RuntimeError("Graph negative control failed")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(run, f"shard-{index}", index * 2, ["--num-shards", "4", "--shard-index", str(index)])
                for index in range(4)
            ]
            for future in concurrent.futures.as_completed(futures):
                state["jobs"].append(future.result())
                save()
        if not all(job["expected_outcome"] for job in state["jobs"]):
            raise RuntimeError("One or more BMM shards failed")
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
