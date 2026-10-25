"""Measure matched native forward calls using the existing performance-gate helpers."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from tools.forward_perf_gate.measurement import measure_cold_and_warm, phase_call

from aisimulate import _runtime
from aisimulate.sdk.inference_session import InferenceSession
from aisimulate_core.sdk import config, perf_database
from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate_core.sdk.config_builders import build_model_config
from aisimulate_core.sdk.fpm_config import OpLevelCompileConfig
from aisimulate_core.sdk.models import get_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", choices=("measured", "anchored"), default="measured")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    model_config = build_model_config(
        tp_size=8,
        pp_size=1,
        attention_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=16,
        gemm_quant_mode="w8a8",
        moe_quant_mode="w8a8_dynamic",
        kvcache_quant_mode="bfloat16",
        system="ascend910c",
    )
    if args.policy == "anchored":
        model_config.op_level_config = OpLevelCompileConfig(
            {"ascend": {"shape_policy": "anchored"}}, system="ascend910c", backend="sglang"
        )
    model = get_model(args.model, model_config, "sglang")
    database = perf_database.get_database_view("ascend910c", "sglang", "0.5.22", database_mode="SILICON")
    session = InferenceSession(model, database, get_backend("sglang"))
    setup_ms = (time.perf_counter() - started) * 1000
    cases = [("context", 1, 1024, 0), ("context", 1, 4096, 0), ("generation", 8, 8192, 0)]
    if args.policy == "anchored":
        cases += [("context", 1, 4096, 32768), ("generation", 8, 65536, 0)]
    records = []
    for phase, batch, length, prefix in cases:
        runtime = config.RuntimeConfig(batch_size=batch, beam_width=1, isl=length + prefix, osl=8, prefix=prefix)
        call = phase_call(session, runtime, phase=phase, stride=32)
        value, cold, samples, summary = measure_cold_and_warm(call, warmup=10, iterations=100)
        records.append(
            {
                "phase": phase,
                "batch": batch,
                "length": length,
                "prefix": prefix,
                "predicted_ms": value,
                "first_call_us": cold,
                "samples_us": samples,
                "warm": summary,
            }
        )
    args.report.write_text(
        json.dumps(
            {
                "policy": args.policy,
                "native_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
                "setup_ms": setup_ms,
                "cases": records,
                "scope": (
                    "Forward gate timing helpers; setup, first query and warm calls separated. "
                    "No end-to-end serving accuracy assertion."
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
