"""Check vector rows through the native operation test interface."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import time

import pyarrow.parquet as parquet

from aisimulate import _runtime
from aisimulate_core.sdk.engine import ENGINE_CONFIG_SCHEMA_VERSION, EngineHandle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[4]
    table = args.table_dir / "ascend_vector_perf.parquet"
    rows = parquet.read_table(table).to_pylist()
    with tempfile.TemporaryDirectory() as temporary:
        systems = Path(temporary)
        shutil.copyfile(repository / "python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml", systems / "ascend910c.yaml")
        destination = systems / "data/ascend910c/vector/sglang/0.5.22"
        destination.mkdir(parents=True)
        shutil.copyfile(table, destination / table.name)
        spec = {"schema_version": _runtime.engine_spec_schema_version(), "engine": {"schema_version": ENGINE_CONFIG_SCHEMA_VERSION, "model_name": "__database_probe__", "system_name": "ascend910c", "backend": "sglang", "backend_version": "0.5.22", "tp_size": 1, "pp_size": 1, "enable_shared_layer": False}, "context_ops": [], "generation_ops": []}
        started = time.perf_counter()
        handle = EngineHandle(_runtime.engine_spec_bincode_from_json(json.dumps(spec)), systems_path=str(systems))
        preparation_ms = (time.perf_counter() - started) * 1000
        durations, maximum_error = [], 0.0
        for row in rows:
            operation = {"AscendVector": {"name": "measured_vector", "scale_factor": 1.0, "key": {"role": row["role"], "dimensions": json.loads(row["dimensions"]), "execution_mode": row["execution_mode"]}, "token_axis": "tokens", "token_split": 1, "token_multiple": 1}}
            assert json.loads(_runtime.op_from_spec_json(json.dumps(operation))._spec_json()) == operation
            started = time.perf_counter()
            values = handle.evaluate_ops_json(json.dumps([operation]), is_context=False, batch_size=row["tokens"], s=8192)
            durations.append((time.perf_counter() - started) * 1000)
            error = abs(values[0][1] - row["latency_ms"])
            assert error <= 1e-12, (row, values)
            maximum_error = max(maximum_error, error)
        report = {"status": "passed", "checked_measured_rows": len(rows), "max_abs_error_ms": maximum_error, "native_binary_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(), "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(), "preparation_ms": preparation_ms, "first_query_ms": durations[0], "subsequent_query_median_ms": statistics.median(durations[1:]), "scope": "Exact measured-row lookup and typed wire; not interpolation/forward/serving accuracy"}
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
