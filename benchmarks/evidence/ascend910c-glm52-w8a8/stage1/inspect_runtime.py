import argparse
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--patch", type=pathlib.Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ("sglang", "torch", "torch-npu", "pydantic", "PyYAML", "psutil", "numpy", "pandas", "scipy", "pyarrow", "ijson", "transformers", "deep-ep"):
        try:
            package = importlib.metadata.distribution(name)
            packages[name] = {"version": package.version, "location": str(package.locate_file(""))}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    versions = {}
    version_read_errors = {}
    for root in (pathlib.Path("/usr/local/Ascend/ascend-toolkit/latest"), pathlib.Path("/usr/local/Ascend/cann"), pathlib.Path("/usr/local/Ascend/nnal")):
        if root.exists():
            candidates = [root / "version.cfg", root / "version.info"]
            candidates.extend(root.glob("*/ascend_toolkit_install.info"))
            candidates.extend(root.glob("*/hccl/version.info"))
            candidates.extend(root.glob("*/hccl/lib64/version.info"))
            candidates.extend(root.glob("share/info/*/version.info"))
            for path in candidates:
                try:
                    if path.is_file():
                        versions[str(path)] = path.read_text(errors="replace")[:12000]
                except OSError as error:
                    version_read_errors[str(path)] = str(error)
    git_sources = []
    for candidate in ("/sgl-workspace/sglang", "/workspace/sglang", "/sglang", "/opt/sglang"):
        path = pathlib.Path(candidate)
        if (path / ".git").exists():
            git_command = ["git", "-c", "safe.directory=" + candidate, "-C", candidate]
            head = subprocess.run([*git_command, "rev-parse", "HEAD"], capture_output=True, text=True)
            status = subprocess.run([*git_command, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
            git_sources.append({"path": candidate, "head": head.stdout.strip(), "tracked_changes": status.stdout.splitlines(), "exit_code": head.returncode, "stderr": head.stderr, "status_exit_code": status.returncode, "status_stderr": status.stderr})
    data = {"python": sys.version, "executable": sys.executable, "packages": packages, "versions": versions, "version_read_errors": version_read_errors, "source_checkouts": git_sources, "device_access": False, "model_loaded": False}
    (args.output / "runtime-image-inventory.json").write_text(json.dumps(data, indent=2) + "\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.repository / "python/aisimulate/src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    commands = [
        ("onboard-help", ["onboard", "--help"]),
        ("onboard-init-help", ["onboard", "init", "--help"]),
        ("onboard-checkpoint-help", ["onboard", "checkpoint", "--help"]),
    ]
    if args.checkpoint.exists():
        commands.append(("onboard-resume", ["onboard", "resume", "--checkpoint", str(args.checkpoint)]))
    else:
        commands.append(("onboard-checkpoint-create", ["onboard", "checkpoint", "--file", str(args.checkpoint)]))
    records = []
    for label, arguments in commands:
        result = subprocess.run([sys.executable, "-m", "aisimulate", *arguments], env=environment, capture_output=True, text=True)
        (args.output / (label + ".stdout.txt")).write_text(result.stdout)
        (args.output / (label + ".stderr.txt")).write_text(result.stderr)
        records.append({"arguments": arguments, "exit_code": result.returncode})
        print(label, result.returncode, flush=True)
        if result.returncode:
            print(result.stderr[-3000:], flush=True)
            break
    if args.patch and all(record["exit_code"] == 0 for record in records):
        saved = json.loads(args.checkpoint.read_text())
        arguments = ["onboard", "checkpoint", "--file", str(args.checkpoint), "--expect-revision", str(saved["revision"]), "--update", str(args.patch)]
        result = subprocess.run([sys.executable, "-m", "aisimulate", *arguments], env=environment, capture_output=True, text=True)
        (args.output / "onboard-checkpoint-update.stdout.txt").write_text(result.stdout)
        (args.output / "onboard-checkpoint-update.stderr.txt").write_text(result.stderr)
        records.append({"arguments": arguments, "exit_code": result.returncode})
        print("checkpoint-update", result.returncode, flush=True)
        if result.returncode:
            print(result.stderr[-3000:], flush=True)
    (args.output / "cli-commands.json").write_text(json.dumps(records, indent=2) + "\n")
    return 0 if all(record["exit_code"] == 0 for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
