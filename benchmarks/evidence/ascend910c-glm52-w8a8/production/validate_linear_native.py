"""Check a measured linear table through the native engine operation interface."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile

import pyarrow.parquet as parquet
import yaml

from aisimulate import _runtime
from aisimulate_core.sdk.engine import ENGINE_CONFIG_SCHEMA_VERSION, EngineHandle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--holdout-plan", type=Path)
    parser.add_argument("--observations", type=Path, nargs="+")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    rows = parquet.read_table(args.table_dir / "ascend_linear_perf.parquet").to_pylist()
    with tempfile.TemporaryDirectory() as temporary:
        systems = Path(temporary)
        spec = yaml.safe_load((root / "python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml").read_text())
        (systems / "ascend910c.yaml").write_text(yaml.safe_dump(spec))
        destination = systems / "data/ascend910c/gemm/sglang/0.5.22"
        destination.mkdir(parents=True)
        shutil.copyfile(args.table_dir / "ascend_linear_perf.parquet", destination / "ascend_linear_perf.parquet")
        engine_spec = {
            "schema_version": _runtime.engine_spec_schema_version(),
            "engine": {
                "schema_version": ENGINE_CONFIG_SCHEMA_VERSION,
                "model_name": "__database_probe__", "system_name": "ascend910c",
                "backend": "sglang", "backend_version": "0.5.22", "tp_size": 1, "pp_size": 1,
                "enable_shared_layer": False,
            },
            "context_ops": [], "generation_ops": [],
        }
        handle = EngineHandle(_runtime.engine_spec_bincode_from_json(json.dumps(engine_spec)), systems_path=str(systems))
        def evaluate(row):
            operation = {"AscendLinear": {
                "name": "measured_linear", "scale_factor": 1.0,
                "key": {field: row[field] for field in ("quant_mode", "n", "k", "weight_format", "execution_mode")},
                "bias": "always" if row["quant_bias"] else "never",
                "token_axis": "tokens", "token_split": 1, "token_multiple": 1,
            }}
            restored = _runtime.op_from_spec_json(json.dumps(operation))
            assert json.loads(restored._spec_json()) == operation
            values = handle.evaluate_ops_json(json.dumps([operation]), is_context=False, batch_size=row["m"], s=8192)
            return values[0][1]

        maximum_error = 0.0
        for row in rows:
            actual = evaluate(row)
            error = abs(actual - row["latency_ms"])
            assert error <= 1e-12, (row, actual)
            maximum_error = max(maximum_error, error)
        print(json.dumps({"status": "passed", "checked_measured_rows": len(rows), "max_abs_error_ms": maximum_error,
                          "native_binary": _runtime.__file__,
                          "native_binary_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
                          "scope": "Measured row lookup/wire correctness; not whole-forward or serving accuracy"}))
        if args.holdout_plan:
            if not args.observations or not args.report:
                parser.error("Holdout validation requires observations and a report path")
            from collector.sglang.ascend_linear_export import validated_rows

            provenance = json.loads((args.table_dir / "provenance.json").read_text())
            identity = provenance["identity"]
            heldout, quality = validated_rows(args.holdout_plan, args.observations, identity["runtime"])
            for key in ("protocol", "serving_setup"):
                if quality["identity"][key] != identity[key]:
                    raise ValueError(f"Holdout differs from published {key}")
            results, groups = [], {}
            for row in heldout:
                actual = evaluate(row)
                error = abs(actual - row["latency_ms"]) / row["latency_ms"]
                entry = {key: row[key] for key in ("quant_mode", "m", "n", "k", "quant_bias", "execution_mode")}
                entry.update(measured_ms=row["latency_ms"], predicted_ms=actual, absolute_relative_error=error)
                results.append(entry)
                groups.setdefault(row["quant_mode"] + "/" + row["execution_mode"], []).append(error)
            summary = {
                key: {"cases": len(values), "median_relative_error": statistics.median(values),
                      "p90_relative_error": sorted(values)[max(0, (9 * len(values) + 9) // 10 - 1)],
                      "max_relative_error": max(values), "over_10_percent": sum(value > 0.1 for value in values)}
                for key, values in sorted(groups.items())
            }
            report = {
                "scope": "Independent intermediate-M measurements; table unchanged; not whole-forward accuracy",
                "table_sha256": hashlib.sha256((args.table_dir / "ascend_linear_perf.parquet").read_bytes()).hexdigest(),
                "quality": quality, "summary": summary, "cases": results,
                "within_10_percent": all(row["absolute_relative_error"] <= 0.1 for row in results),
            }
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"summary": summary, "within_10_percent": report["within_10_percent"]}))


if __name__ == "__main__":
    main()
