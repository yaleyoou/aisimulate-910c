"""Drive source-state observations and release only the owned recorder container."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="vectors-20261023")
    args = parser.parse_args()
    if not args.suffix or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in args.suffix):
        parser.error("Use a simple campaign suffix")
    os.umask(0o022)
    production = Path(__file__).resolve().parent
    repository = production.parents[3]
    output = production / ("operator-input-state-node32-20261022-" + args.suffix)
    container = "aisim-current-inputs-20261022-" + args.suffix
    state_path = output / "request-campaign.json"
    if state_path.exists():
        raise ValueError("Preserve the existing request campaign")
    state = {"status": "waiting_for_health", "container": container, "cases": [], "scope": "Operator/routing metadata and instrumented startup memory; not performance"}
    state["request_source_sha256"] = {}
    for name in ("ascend_state_requests.py", "ascend_state_sweep.py"):
        source = repository / "python/aisimulate/collector/sglang" / name
        shutil.copyfile(source, output / "source" / name)
        state["request_source_sha256"][name] = hashlib.sha256(source.read_bytes()).hexdigest()

    def save():
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    def run(command, name):
        state.update(status="collecting", active_case=name)
        save()
        with (output / (name + ".log")).open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        state["cases"].append({"case": name, "command": command, "returncode": result.returncode})
        save()
        result.check_returncode()

    save()
    try:
        deadline = time.monotonic() + 1200
        while True:
            running = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Running}}", container], text=True).strip()
            if running != "true":
                raise RuntimeError("Owned recorder exited before becoming healthy")
            try:
                with urllib.request.urlopen("http://127.0.0.1:18000/health", timeout=30) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError, socket.timeout):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Owned recorder did not become healthy")
            time.sleep(5)
        capacities = []
        for path in sorted((output / "memory").glob("rank-*.jsonl")):
            records = [json.loads(line) for line in path.read_text().splitlines()]
            capacities.append(next(record["max_total_num_tokens"] for record in records if record["boundary"] == "alloc_memory_pool:after"))
        if len(capacities) != 16:
            raise ValueError("Incomplete per-rank KV capacity observations")
        state["kv_tokens_per_dp"] = min(capacities)
        prefix = ["python3", str(output / "source/ascend_state_requests.py"), "--capture-dir", str(output / "capture")]
        run(prefix + ["--case", "vectors8192_c1", "--input-tokens", "8192", "--output-tokens", "16", "--max-forwards", "3"], "vectors8192_c1")
        for name, concurrency, inputs, from_start, forwards in (
            ("vectors8192_c4", 4, 8192, True, 6),
            ("vectors128_c18", 18, 128, False, 3),
            ("vectors128_c34", 34, 128, False, 3),
            ("vectors128_c50", 50, 128, False, 3),
        ):
            command = ["python3", str(output / "source/ascend_state_sweep.py"), "--capture-dir", str(output / "capture"), "--case", name, "--concurrency", str(concurrency), "--input-tokens", str(inputs), "--output-tokens", "64", "--arm-after-tokens", "4", "--max-forwards", str(forwards), "--max-running-requests", "64", "--kv-budget-per-dp", str(state["kv_tokens_per_dp"])]
            if from_start:
                command.append("--capture-from-start")
            run(command, name)
        state["status"] = "completed"
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        with (output / "startup.log").open("w") as stream:
            subprocess.run(["docker", "logs", container], stdout=stream, stderr=subprocess.STDOUT, check=False)
        subprocess.run(["docker", "stop", "--time", "30", container], check=False)
        for path in (output, *output.rglob("*")):
            os.chown(path, 5031, 5031)


if __name__ == "__main__":
    main()
