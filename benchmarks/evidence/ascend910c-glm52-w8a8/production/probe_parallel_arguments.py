"""Resolve actual SGLang parallel settings without loading model weights."""

import argparse
import hashlib
import json
import subprocess
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.case is not None:
        from sglang.srt.hardware_backend.npu.utils import init_npu_backend
        from sglang.srt.server_args import ServerArgs

        init_npu_backend()
        case = json.loads(args.case)
        try:
            server = ServerArgs(
                model_path=str(args.model),
                device="npu",
                dtype="bfloat16",
                quantization="modelslim",
                context_length=180000,
                disable_shared_experts_fusion=True,
                **case,
            )
            server.resolve_once()
            resolved = server.resolved_dict()
            fields = (
                "tp_size",
                "attn_dp_size",
                "dp_size",
                "pp_size",
                "ep_size",
                "attn_cp_size",
                "dcp_size",
                "moe_a2a_backend",
                "moe_runner_backend",
                "moe_dense_tp_size",
                "cp_strategy",
                "disable_overlap_schedule",
                "enable_dp_lm_head",
                "page_size",
                "dcp_comm_backend",
            )
            result = {
                "status": "arguments_resolved",
                "requested": case,
                "resolved": {key: resolved.get(key) for key in fields},
            }
        except Exception:
            result = {"status": "arguments_rejected", "requested": case, "error": traceback.format_exc()}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    args.output.mkdir(exist_ok=False)
    cases = []
    for attention_dp in (1, 2, 4, 8, 16):
        cases.append({"tp_size": 16, "attn_dp_size": attention_dp, "ep_size": 16, "moe_a2a_backend": "deepep"})
    for experts in (1, 2, 4, 8):
        cases.append({"tp_size": 16, "attn_dp_size": 1, "ep_size": experts, "moe_a2a_backend": "none"})
    for context in (2, 4, 8):
        cases.append(
            {
                "tp_size": 16,
                "attn_dp_size": 2,
                "attn_cp_size": context,
                "enable_prefill_cp": True,
                "cp_strategy": "interleave",
                "ep_size": 16,
                "moe_a2a_backend": "deepep",
            }
        )
        cases.append(
            {"tp_size": 16, "attn_dp_size": 2, "dcp_size": context, "ep_size": 16, "moe_a2a_backend": "deepep"}
        )
    for pipeline in (2, 4):
        cases.append(
            {
                "tp_size": 16 // pipeline,
                "pp_size": pipeline,
                "attn_dp_size": 1,
                "ep_size": 16 // pipeline,
                "moe_a2a_backend": "deepep",
            }
        )
    results = []
    for index, case in enumerate(cases):
        destination = args.output / f"case-{index:02}.json"
        command = [
            "python3",
            str(Path(__file__).resolve()),
            "--model",
            str(args.model),
            "--case",
            json.dumps(case),
            "--output",
            str(destination),
        ]
        with (args.output / f"case-{index:02}.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
        if result.returncode != 0 or not destination.is_file():
            raise RuntimeError(f"Probe infrastructure failed for case{index}")
        results.append(json.loads(destination.read_text()))
    revision = subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip()
    summary = {
        "scope": "argument validation only; no distributed construction or kernel execution",
        "framework_commit": revision,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
