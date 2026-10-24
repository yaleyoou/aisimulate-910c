"""Launch a verified current-model view for bounded operator-input observations."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--capture-vectors", action="store_true")
    parser.add_argument("--capture-memory", action="store_true")
    parser.add_argument("--capture-boundaries", action="store_true")
    parser.add_argument("--attention-dp", type=int, choices=(1, 2, 4, 8, 16), default=2)
    parser.add_argument("--pipeline", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--expert-parallel", type=int)
    parser.add_argument("--moe-a2a", choices=("deepep", "none"), default="deepep")
    parser.add_argument("--prefill-cp", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--decode-cp", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--decode-graph-buckets", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--node", choices=("npu1-31", "npu1-32"), default="npu1-31")
    args = parser.parse_args()
    tensor_world = 16 // args.pipeline
    expert_parallel = args.expert_parallel or tensor_world
    if tensor_world % args.attention_dp or expert_parallel < 1 or tensor_world % expert_parallel:
        parser.error("Attention DP and EP must divide the per-stage tensor group")
    if tensor_world % (args.attention_dp * args.prefill_cp):
        parser.error("Prefill CP must divide the attention group")
    if not 0 < args.mem_fraction_static < 1 or args.prefill_chunk < 1:
        parser.error("Use a valid memory fraction and positive per-replica prefill chunk")
    if args.max_total_tokens is not None and args.max_total_tokens < 1:
        parser.error("KV token limit must be positive")
    if not re.fullmatch(r"[a-z0-9-]*", args.suffix):
        parser.error("Use a simple suffix")
    if args.decode_graph_buckets != sorted(set(args.decode_graph_buckets)) or min(args.decode_graph_buckets) < 1:
        parser.error("Decode graph buckets must be sorted unique positive sizes")
    suffix = "-" + args.suffix if args.suffix else ""
    if socket.gethostname() != args.node:
        raise RuntimeError("Collection host differs from the requested node")
    production = Path(__file__).resolve().parent
    repository = production.parents[3]
    view = json.loads((production / "model-view-plan.json").read_text())
    copied = json.loads((production / "model-overrides-copy.json").read_text())
    if view["status"] != "verified_readonly_mount_plan" or copied["status"] != "completed":
        raise ValueError("Current model view is not verified")
    overrides = view["overrides"] if args.node == "npu1-31" else []
    checked_files = []
    if args.node == "npu1-32":
        fingerprint = json.loads((production / "model-current-node32-fingerprint.json").read_text())
        if fingerprint["checkpoint_sha256"] != view["target_checkpoint_sha256"]:
            raise ValueError("Current model identities differ")
        for item in fingerprint["files"]:
            if Path(item["path"]).suffix == ".json":
                path = Path(fingerprint["model_path"]) / item["path"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError(f"Current model metadata changed: {path.name}")
                checked_files.append(item["path"])
    output = production / ("operator-input-state-" + args.node.replace("npu1-", "node") + "-20261022" + suffix)
    output.mkdir(exist_ok=False)
    (output / "source").mkdir()
    (output / "capture").mkdir()
    before = subprocess.check_output(["npu-smi", "info"], text=True)
    (output / "npu-before.log").write_text(before)
    if before.count("No running processes found in NPU") != 8:
        raise RuntimeError("NPU process inventory is not empty")
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", 18000))
    source_hashes = {}
    names = ["ascend_state_capture.py", "ascend_input_capture.py", "ascend_recording_server.py"]
    if args.capture_memory:
        names.append("ascend_memory_capture.py")
    for name in names:
        source = repository / "python/aisimulate/collector/sglang" / name
        shutil.copyfile(source, output / "source" / name)
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    setup = json.loads((production / "serving-setup.json").read_text())
    container = "aisim-current-inputs-20261022" + suffix
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
        "32g",
        "--mount",
        "type=bind,src=/mnt/raid/user/data/models/glm-q,dst=/models/glm-q,readonly",
        "--mount",
        f"type=bind,src={output},dst=/campaign",
        "-e",
        "AISIM_CAPTURE_DIR=/campaign/capture",
        "-e",
        "AISIM_CAPTURE_INPUTS=1",
        "-e",
        "HCCL_IF_BASE_PORT=39000",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "MKL_NUM_THREADS=1",
    ]
    if args.capture_vectors:
        command.extend(("-e", "AISIM_CAPTURE_VECTORS=1"))
    if args.capture_memory:
        command.extend(("-e", "AISIM_MEMORY_DIR=/campaign/memory"))
    if args.capture_boundaries:
        command.extend(("-e", "AISIM_CAPTURE_BOUNDARIES=1"))
    for override in overrides:
        command.extend(("--mount", f"type=bind,src={override['source']},dst={override['destination']},readonly"))
    for path in (
        "/usr/local/Ascend/driver",
        "/usr/local/Ascend/firmware",
        "/usr/local/sbin",
        "/etc/ascend_install.info",
    ):
        command.extend(("--mount", f"type=bind,src={path},dst={path},readonly"))
    for name, value in setup["environment"].items():
        if value is not None:
            command.extend(("-e", f"{name}={value}"))
    command.extend(
        (
            "--entrypoint",
            "python3",
            "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443",
            "/campaign/source/ascend_recording_server.py",
        )
    )
    command.extend(
        [
            "--model-path",
            "/models/glm-q",
            "--host",
            "0.0.0.0",
            "--port",
            "18000",
            "--context-length",
            "180000",
            "--nnodes",
            "1",
            "--node-rank",
            "0",
            "--tp-size",
            str(tensor_world),
            "--pp-size",
            str(args.pipeline),
            "--dp-size",
            str(args.attention_dp),
            "--ep-size",
            str(expert_parallel),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
            "--max-running-requests",
            "64",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--quantization",
            "modelslim",
            "--dtype",
            "bfloat16",
            "--served-model-name",
            "glm-5-aisim",
            "--chunked-prefill-size",
            str(args.prefill_chunk * args.attention_dp),
            "--max-prefill-tokens",
            "32768",
            "--disable-shared-experts-fusion",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--moe-a2a-backend",
            args.moe_a2a,
            "--deepep-mode",
            "auto",
            "--enable-cache-report",
            "--load-balance-method",
            "round_robin",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load":false}',
            "--weight-loader-disable-mmap",
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-max-bs-decode",
            str(max(args.decode_graph_buckets)),
            "--cuda-graph-bs-decode",
            *map(str, args.decode_graph_buckets),
        ]
    )
    if args.prefill_cp > 1:
        command.extend(("--enable-prefill-cp", "--cp-strategy", "interleave", "--attn-cp-size", str(args.prefill_cp)))
    if args.decode_cp > 1:
        command.extend(("--decode-context-parallel-size", str(args.decode_cp)))
    if args.max_total_tokens is not None:
        command.extend(("--max-total-tokens", str(args.max_total_tokens)))
    result = subprocess.run(command, capture_output=True, text=True)
    record = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "checkpoint_sha256": view["target_checkpoint_sha256"],
        "source_sha256": source_hashes,
        "node": args.node,
        "checked_model_metadata": checked_files,
        "scope": "Instrumented operator/routing state and optional memory; graph memory can include observer allocations; not performance",
    }
    (output / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    result.check_returncode()
    print(json.dumps({"container": container, "port": 18000, "checkpoint_sha256": view["target_checkpoint_sha256"]}))


if __name__ == "__main__":
    main()
