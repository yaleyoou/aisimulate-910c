import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet

from aisimulate import _runtime
from aisimulate_core.sdk.engine import ENGINE_CONFIG_SCHEMA_VERSION, EngineHandle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[4]
    systems = repository / "python/aisimulate/src/aisimulate_core/systems"
    table = systems / "data/ascend910c/attention/sglang/0.5.22/ascend_attention_perf.parquet"
    spec = {
        "schema_version": _runtime.engine_spec_schema_version(),
        "engine": {
            "schema_version": ENGINE_CONFIG_SCHEMA_VERSION,
            "model_name": "__attention_publication_integrity__",
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
    handle = EngineHandle(_runtime.engine_spec_bincode_from_json(json.dumps(spec)), systems_path=str(systems))
    groups, maximum_error = Counter(), 0.0
    rows = parquet.read_table(table).to_pylist()
    for row in rows:
        context = row["phase"] == "prefill"
        operation = {
            "AscendAttention": {
                "name": "publication_integrity",
                "scale_factor": 1.0,
                "key": {name: row[name] for name in ("operation", "phase", "local_heads", "execution_mode")},
                "graph_buckets": [row["padded_batch"]] if not context and row["execution_mode"] == "graph" else [],
            }
        }
        predicted = handle.evaluate_ops_json(
            json.dumps([operation]),
            is_context=context,
            batch_size=row["active_batch"],
            s=row["tokens"] // row["active_batch"] if context else row["sequence_length"],
            prefix=row["prefix"],
        )[0][1]
        maximum_error = max(maximum_error, abs(predicted - row["latency_ms"]))
        assert abs(predicted - row["latency_ms"]) < 1e-9, row
        groups[f"{row['operation']}/{row['phase']}/{row['local_heads']}/{row['execution_mode']}"] += 1
    components = []
    for operation in ("dispatch", "combine"):
        components.append(
            {
                "AscendDeepEp": {
                    "name": "decode_moe_" + operation,
                    "scale_factor": 75.0,
                    "key": {
                        "phase": "low_latency",
                        "operation": operation,
                        "num_ranks": 16,
                        "ranks_per_node": 16,
                        "distribution": "balanced",
                        "execution_mode": "graph",
                    },
                    "token_split": 8,
                    "input_token_multiple": 8,
                }
            }
        )
    components.append(
        {
            "AscendMoeChain": {
                "name": "decode_expert_compute",
                "scale_factor": 75.0,
                "key": {"phase": "decode", "local_experts": 16, "distribution": "balanced", "execution_mode": "graph"},
                "token_numerator": 16,
                "token_denominator": 16,
                "input_token_multiple": 8,
            }
        }
    )
    component_results = handle.evaluate_ops_json(json.dumps(components), is_context=False, batch_size=1, s=8192)
    result = {
        "scope": "Native exact-point and diagnostic component queries; "
        "full request results use the canonical predict CLI.",
        "rows": len(rows),
        "maximum_absolute_error_ms": maximum_error,
        "groups": dict(groups),
        "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
        "tp8_adp2_ep16_decode_batch1_components_75_layers": component_results,
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
