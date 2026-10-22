"""Run frozen linear controls or a full grid on a dedicated calibration container."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


IMAGE = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--mode", choices=("controls", "full"), required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[16, 32, 32, 16])
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    os.umask(0o022)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    production = Path(__file__).resolve().parent
    repository = production.parents[3]
    sources = {
        "collector.py": repository / "python/aisimulate/collector/sglang/ascend_linear.py",
        "plan.json": args.plan or production / "linear-20261022/plan-v2.json",
        "serving-setup-nodes.json": production / "serving-setup.json",
        "graph_noop.py": production / "hccl_graph_noop_probe.py",
    }
    for name, source in sources.items():
        shutil.copyfile(source, root / name)
    state = {
        "mode": args.mode,
        "status": "preflight",
        "image": IMAGE,
        "host": os.uname().nodename,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        "jobs": [],
    }

    def save():
        (root / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")

    for name, command in {
        "npu-before": ["npu-smi", "info"],
        "containers-before": ["docker", "ps", "--format", "{{.Names}} {{.Status}}"],
        "disk-before": ["df", "-h", str(root), "/dev/shm"],
        "driver": ["cat", "/usr/local/Ascend/driver/version.info"],
    }.items():
        (root / (name + ".log")).write_text(subprocess.check_output(command, text=True))
    command = [
        "docker", "run", "-d", "--name", args.container, "--network", "host", "--privileged", "--shm-size", "32g",
        "--mount", f"type=bind,src={root},dst=/campaign",
        "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
    ]
    for path in ("/usr/local/Ascend/driver", "/usr/local/Ascend/firmware", "/usr/local/sbin", "/etc/ascend_install.info"):
        if Path(path).exists():
            command.extend(("--mount", f"type=bind,src={path},dst={path},readonly"))
    command.extend(("--entrypoint", "/bin/sleep", IMAGE, "infinity"))
    state["create_command"] = command
    save()
    subprocess.run(command, check=True)

    def run(name, device, extra, negative=False):
        arguments = ["python3"]
        if negative:
            arguments.append("/campaign/graph_noop.py")
        arguments.extend(("/campaign/collector.py", "--plan", "/campaign/plan.json", "--serving-setup",
                          "/campaign/serving-setup-nodes.json", "--device", str(device), "--output",
                          f"/campaign/{name}.json", *extra))
        command = ["docker", "exec", args.container, "timeout", "7200", *arguments]
        started = time.time()
        with (root / (name + ".log")).open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        path = root / (name + ".json")
        report = json.loads(path.read_text()) if path.exists() else {}
        expected = (
            result.returncode != 0 and report.get("status") == "failed"
            and "numerical check failed" in report.get("error", "")
            if negative else result.returncode == 0 and report.get("status") == "completed"
        )
        return {"name": name, "command": command, "returncode": result.returncode,
                "expected_outcome": expected, "cases": len(report.get("cases", [])),
                "elapsed_seconds": time.time() - started}

    try:
        state["status"] = "running"
        save()
        if args.mode == "controls":
            for index, copies in enumerate(args.copies):
                job = run(f"m8-{index}-copies{copies}", 0,
                          ["--case-filter", '"m": 8,', "--weight-copies", str(copies)])
                state["jobs"].append(job)
                save()
                if not job["expected_outcome"]:
                    raise RuntimeError(job)
            for name, device, extra, negative in (
                ("large16384", 2, ["--case-filter", '"m": 16384,', "--weight-copies", str(max(args.copies))], False),
                ("graph-noop", 4, ["--case-filter", '"execution_mode": "graph"'], True),
            ):
                job = run(name, device, extra, negative)
                state["jobs"].append(job)
                save()
                if not job["expected_outcome"]:
                    raise RuntimeError(job)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(run, f"shard-{index}", index * 2,
                                       ["--num-shards", "4", "--shard-index", str(index)]) for index in range(4)]
                for future in concurrent.futures.as_completed(futures):
                    state["jobs"].append(future.result())
                    save()
            if not all(job["expected_outcome"] for job in state["jobs"]):
                raise RuntimeError("One or more shards failed; preserve raw observations")
        state["status"] = "completed"
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
        subprocess.run(["docker", "stop", "--time", "30", args.container], check=False)
        for path in (root, *root.rglob("*")):
            os.chown(path, 5031, 5031)


if __name__ == "__main__":
    main()
