"""Run the four frozen PD attention-TP mappings sequentially on one node."""

import argparse
import json
from pathlib import Path
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    production = Path(__file__).resolve().parent
    output = production / ("pd-grid-" + socket.gethostname() + "-" + args.suffix + ".json")
    if output.exists():
        parser.error("Use a fresh grid suffix")
    state = {"status": "running", "jobs": []}
    output.write_text(json.dumps(state, indent=2) + "\n")
    try:
        for index, (source, destination) in enumerate(((8, 8), (4, 8), (8, 4), (4, 4))):
            deadline = time.monotonic() + 60
            while subprocess.check_output(["npu-smi", "info"], text=True).count("No running processes found in NPU") != 8:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Node did not become idle after the previous owned job")
                time.sleep(1)
            command = [
                "python3", str(production / "run_pd_probe.py"), "--node-rank", str(args.node_rank),
                "--suffix", f"pd{source}{destination}-{args.suffix}", "--plan", str(args.plan.resolve()),
                "--source-tp", str(source), "--destination-tp", str(destination),
                "--master-port", str(30300 + index * 10), "--store-port", str(30400 + index * 10),
            ]
            result = subprocess.run(command)
            state["jobs"].append({"command": command, "returncode": result.returncode})
            output.write_text(json.dumps(state, indent=2) + "\n")
            result.check_returncode()
        state["status"] = "completed"
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        output.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
