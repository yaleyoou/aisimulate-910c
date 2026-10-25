"""Check every qualified DCP storage/attention point through the native API."""

import argparse
import hashlib
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as parquet

from aisimulate import _runtime
from aisimulate_core.sdk.engine import ENGINE_CONFIG_SCHEMA_VERSION, EngineHandle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shape-policy", choices=("measured", "anchored"), default="measured")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[4]
    table = args.table_dir / "ascend_dcp_attention_perf.parquet"
    rows = parquet.read_table(table).to_pylist()
    with tempfile.TemporaryDirectory() as temporary:
        systems = Path(temporary)
        shutil.copyfile(
            repository / "python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml", systems / "ascend910c.yaml"
        )
        destination = systems / "data/ascend910c/attention/sglang/0.5.22"
        destination.mkdir(parents=True)
        shutil.copyfile(table, destination / table.name)
        spec = {
            "schema_version": _runtime.engine_spec_schema_version(),
            "engine": {
                "schema_version": ENGINE_CONFIG_SCHEMA_VERSION,
                "model_name": "__database_probe__",
                "system_name": "ascend910c",
                "backend": "sglang",
                "backend_version": "0.5.22",
                "tp_size": 1,
                "pp_size": 1,
                "enable_shared_layer": False,
            },
            "context_ops": [],
            "generation_ops": [],
        }
        spec["engine"]["extra"] = {
            "estimator_config": json.dumps({"op_level": {"ascend": {"shape_policy": args.shape_policy}}})
        }
        started = time.perf_counter()
        handle = EngineHandle(_runtime.engine_spec_bincode_from_json(json.dumps(spec)), systems_path=str(systems))
        preparation_ms = (time.perf_counter() - started) * 1000
        durations, maximum = [], 0.0
        for row in rows:
            context = row["phase"] == "prefill"
            operation = {
                "AscendDcpAttention": {
                    "name": "measured_dcp_storage_attention",
                    "scale_factor": 1.0,
                    "key": {
                        field: row[field]
                        for field in ("phase", "local_heads", "dcp_size", "remap_topk", "execution_mode")
                    },
                    "graph_buckets": [] if context else [row["padded_batch"]],
                }
            }
            assert json.loads(_runtime.op_from_spec_json(json.dumps(operation))._spec_json()) == operation
            started = time.perf_counter()
            values = handle.evaluate_ops_json(
                json.dumps([operation]),
                is_context=context,
                batch_size=row["active_batch"],
                s=row["query_tokens_per_request"] if context else row["sequence_length"],
                prefix=row["prefix"],
            )
            durations.append((time.perf_counter() - started) * 1000)
            difference = abs(values[0][1] - row["latency_ms"])
            assert difference <= 1e-12, (row, values)
            maximum = max(maximum, difference)
        report = {
            "status": "passed",
            "checked_measured_rows": len(rows),
            "max_abs_error_ms": maximum,
            "native_binary_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
            "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
            "preparation_ms": preparation_ms,
            "first_query_ms": durations[0],
            "warm_query_median_ms": statistics.median(durations[1:]),
            "scope": "Exact measured point and typed wire validation; not serving accuracy",
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
