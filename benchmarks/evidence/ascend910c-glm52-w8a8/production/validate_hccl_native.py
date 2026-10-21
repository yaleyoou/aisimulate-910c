"""Check collected HCCL rows through the existing native engine/spec interface."""

import argparse
import json
import pathlib
import shutil
import tempfile

import pyarrow.parquet as parquet
import yaml

import aisimulate

aisimulate.__path__.append("/tmp/aisim-native")
from aisimulate import _runtime
from aisimulate_core.sdk.engine import ENGINE_CONFIG_SCHEMA_VERSION, EngineHandle


def main():
    root = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", type=pathlib.Path, default=root / "hccl-peer-qualified-20261022")
    args = parser.parse_args()
    table_root = args.table_dir
    rows = parquet.read_table(table_root / "hccl_perf.parquet").to_pylist()
    with tempfile.TemporaryDirectory() as temporary:
        systems = pathlib.Path(temporary)
        spec = yaml.safe_load((root.parents[3] / "python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml").read_text())
        spec["misc"]["hccl_version"] = "peer-test"
        (systems / "ascend910c.yaml").write_text(yaml.safe_dump(spec))
        destination = systems / "data/ascend910c/comm/hccl/peer-test"
        destination.mkdir(parents=True)
        shutil.copyfile(table_root / "hccl_perf.parquet", destination / "hccl_perf.parquet")
        (systems / "data/ascend910c/sglang/0.5.22").mkdir(parents=True)
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
        checked = 0
        for row in rows:
            element_bytes = 2 if row["dtype"] in {"bfloat16", "float16"} else 4
            operation = {"Hccl": {
                "name": "collected_hccl", "scale_factor": 1.0,
                "key": {field: row[field] for field in ("dtype", "operation", "num_ranks", "ranks_per_node", "rank_layout", "execution_mode")},
                "elements_per_token": row["input_bytes_per_rank"] // element_bytes, "token_split": 1,
            }}
            restored = _runtime.op_from_spec_json(json.dumps(operation))
            assert json.loads(restored._spec_json()) == operation
            values = handle.evaluate_ops_json(json.dumps([operation]), is_context=False, batch_size=1, s=8192)
            actual = values[0][1]
            assert abs(actual - row["latency_ms"]) <= 1e-12, (operation, actual, row["latency_ms"])
            checked += 1
        print(json.dumps({"checked_measured_rows": checked, "status": "passed", "native_binary": _runtime.__file__}))


if __name__ == "__main__":
    main()
