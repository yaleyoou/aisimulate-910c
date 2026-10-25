"""Compare native anchored estimates with an independent NPU attention campaign."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as parquet

from aisimulate import _runtime
from aisimulate_core.sdk.engine import ENGINE_CONFIG_SCHEMA_VERSION, EngineHandle
from aisimulate_core.sdk.errors import PerfDataNotAvailableError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-table", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    systems = Path(__file__).resolve().parents[4] / "python/aisimulate/src/aisimulate_core/systems"
    spec = {
        "schema_version": _runtime.engine_spec_schema_version(),
        "engine": {
            "schema_version": ENGINE_CONFIG_SCHEMA_VERSION,
            "model_name": "__independent_shape_validation__",
            "system_name": "ascend910c",
            "backend": "sglang",
            "backend_version": "0.5.22",
            "tp_size": 1,
            "pp_size": 1,
            "enable_shared_layer": False,
            "extra": {"estimator_config": json.dumps({"op_level": {"ascend": {"shape_policy": "anchored"}}})},
        },
        "context_ops": [],
        "generation_ops": [],
    }
    anchored = EngineHandle(_runtime.engine_spec_bincode_from_json(json.dumps(spec)), systems_path=str(systems))
    spec["engine"]["extra"] = {}
    measured = EngineHandle(_runtime.engine_spec_bincode_from_json(json.dumps(spec)), systems_path=str(systems))
    rows = parquet.read_table(args.validation_table).to_pylist()
    records, groups = [], defaultdict(list)
    for row in rows:
        context = row["phase"] == "prefill"
        operation = {
            "AscendAttention": {
                "name": "independent_shape_probe",
                "scale_factor": 1.0,
                "key": {field: row[field] for field in ("operation", "phase", "local_heads", "execution_mode")},
                "graph_buckets": [] if context else [row["padded_batch"]],
            }
        }
        arguments = {
            "is_context": context,
            "batch_size": row["active_batch"],
            "s": row["tokens"] // row["active_batch"] if context else row["sequence_length"],
            "prefix": row["prefix"],
        }
        predicted = anchored.evaluate_ops_json(json.dumps([operation]), **arguments)[0][1]
        assert math.isfinite(predicted) and predicted > 0
        try:
            baseline = measured.evaluate_ops_json(json.dumps([operation]), **arguments)[0][1]
        except PerfDataNotAvailableError:
            baseline = None
        if baseline is not None:
            assert baseline == predicted
        relative_error = (predicted - row["latency_ms"]) / row["latency_ms"]
        record = {
            "operation": row["operation"],
            "phase": row["phase"],
            "local_heads": row["local_heads"],
            "batch": row["active_batch"],
            "sequence": row["sequence_length"],
            "prefix": row["prefix"],
            "measured_ms": row["latency_ms"],
            "predicted_ms": predicted,
            "relative_error": relative_error,
            "newly_supported": baseline is None,
        }
        records.append(record)
        groups[row["operation"], row["phase"]].append(abs(relative_error))
    report = {
        "status": "completed",
        "scope": (
            "Independent point comparison, not full-serving accuracy acceptance. "
            "No validation rows inserted into calibration tables."
        ),
        "native_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
        "calibration_table_sha256": hashlib.sha256(
            (systems / "data/ascend910c/attention/sglang/0.5.22/ascend_attention_perf.parquet").read_bytes()
        ).hexdigest(),
        "validation_table_sha256": hashlib.sha256(args.validation_table.read_bytes()).hexdigest(),
        "cases": records,
        "groups": [
            {
                "operation": key[0],
                "phase": key[1],
                "cases": len(values),
                "median_absolute_relative_error": statistics.median(values),
                "max_absolute_relative_error": max(values),
            }
            for key, values in groups.items()
        ],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(records),
                "newly_supported": sum(row["newly_supported"] for row in records),
                "groups": report["groups"],
            }
        )
    )


if __name__ == "__main__":
    main()
