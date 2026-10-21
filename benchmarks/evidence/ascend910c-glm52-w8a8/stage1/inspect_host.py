import argparse
import csv
import datetime
import hashlib
import json
import pathlib
import re
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    commands = {
        "npu-status": ["npu-smi", "info"],
        "npu-mapping": ["npu-smi", "info", "-m"],
        "npu-topology": ["npu-smi", "info", "-t", "topo"],
        "npu-board": ["npu-smi", "info", "-t", "board", "-i", "0"],
        "npu-product": ["npu-smi", "info", "-t", "product", "-i", "0", "-c", "0"],
        "npu-memory": ["npu-smi", "info", "-t", "memory", "-i", "0", "-c", "0"],
    }
    observations = {}
    for label, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        text = re.sub(r"(?m)^.*Serial Number.*$", "Serial Number: omitted", result.stdout)
        (args.output / (label + ".txt")).write_text(text + result.stderr)
        observations[label] = {"command": command, "exit_code": result.returncode, "output": label + ".txt"}
        print(label, result.returncode, flush=True)
    files = {}
    sources = [
        pathlib.Path("/usr/local/Ascend/driver/version.info"),
        pathlib.Path("/usr/local/Ascend/cann/share/info/hccl/version.info"),
    ]
    for directory in (pathlib.Path("/home/youxingye/United deployment/service"), pathlib.Path("/home/youxingye/glm52-pd-1p1d")):
        sources.extend(sorted(directory.glob("*.sh")))
    history = pathlib.Path("/home/youxingye/United deployment/results/workload-concurrency-sweep_8k_1k")
    sources.extend([history / "summary.tsv", history / "sweep-config.env", history / "status.tsv"])
    for path in sources:
        content = path.read_bytes()
        files[str(path)] = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    historical_config = dict(line.split("=", 1) for line in (history / "sweep-config.env").read_text().splitlines() if "=" in line)
    current_script = pathlib.Path("/home/youxingye/United deployment/service/united-config.sh").read_text()
    current_defaults = {}
    for match in re.finditer(r"(?m)^(\w+)=\$\{\1:-([^}]*)\}", current_script):
        current_defaults[match.group(1)] = match.group(2).strip('"')
    report = {
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commands": observations, "sources": files,
        "driver_version_file": pathlib.Path("/usr/local/Ascend/driver/version.info").read_text(),
        "host_hccl_version_file": pathlib.Path("/usr/local/Ascend/cann/share/info/hccl/version.info").read_text(),
        "united_current_script_defaults": current_defaults,
        "historical_benchmark_configuration": historical_config,
        "historical_8k1k_results": list(csv.DictReader((history / "summary.tsv").open(), delimiter="\t")),
        "historical_limits": [
            "Historical files are evidence of a prior run, not current startup verification.",
            "Image tag in historical config does not prove the current image digest was used then.",
            "No immutable full checkpoint revision is present in this benchmark configuration.",
            "Current script defaults may differ from the effective settings of the historical server.",
        ],
    }
    (args.output / "host-and-deployment-inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
