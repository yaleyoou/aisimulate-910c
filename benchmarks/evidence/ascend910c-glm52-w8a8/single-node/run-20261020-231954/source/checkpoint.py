import argparse
import json
import pathlib
import subprocess

from run import IMAGE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=pathlib.Path)
    parser.add_argument("--log", type=pathlib.Path, required=True)
    args = parser.parse_args()
    campaign = pathlib.Path(__file__).resolve().parent
    evidence = campaign.parent
    repository = evidence.parents[2]
    checkpoint = evidence / "onboarding-checkpoint.json"
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "PYTHONPATH=/cli-deps:" + str(repository / "python/aisimulate/src"),
        "--mount", "type=bind,src=/tmp/aisim-stage1-cli-deps-20261020,dst=/cli-deps,readonly",
        "--mount", "type=bind,src=" + str(repository) + ",dst=" + str(repository) + ",readonly",
        "--mount", "type=bind,src=" + str(evidence) + ",dst=" + str(evidence),
        "--mount", "type=bind,src=/home/youxingye/United deployment/service,dst=/home/youxingye/United deployment/service,readonly",
        "--entrypoint", "python3", IMAGE, "-m", "aisimulate", "onboard",
    ]
    resumed = subprocess.run(command + ["resume", "--checkpoint", str(checkpoint)], capture_output=True, text=True, check=True)
    data = json.loads(resumed.stdout)
    if data["integrity_issues"]:
        raise RuntimeError(data["integrity_issues"])
    if args.patch:
        updated = subprocess.run(command + ["checkpoint", "--file", str(checkpoint), "--expect-revision", str(data["state"]["revision"]), "--update", str(args.patch.resolve())], capture_output=True, text=True)
        args.log.write_text(updated.stdout + updated.stderr)
        updated.check_returncode()
        data = json.loads(updated.stdout)
    else:
        args.log.write_text(resumed.stdout)
    print(json.dumps({"revision": data["state"]["revision"], "integrity_issues": data["integrity_issues"]}))


if __name__ == "__main__":
    main()
