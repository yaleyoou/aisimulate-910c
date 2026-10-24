import argparse
import json
from pathlib import Path

import pyarrow.parquet as parquet
import yaml

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(16, 32), default=32)
    args = parser.parse_args()
    raw = yaml.safe_load((args.repo / "examples/cli/ascend910c-glm52-pd-8k1k.yaml").read_text())
    raw["engine"]["model"] = args.model
    table = args.repo / "python/aisimulate/src/aisimulate_core/systems/data/ascend910c"
    rows = parquet.read_table(table / "kv_transfer/sglang/0.5.22/ascend_pd_perf.parquet").to_pylist()
    models = {}
    results = []
    for row in rows:
        key = row["source_tp"], row["destination_tp"], row["page_layout"]
        if key not in models:
            raw["engine"]["estimator_config"]["op_level"]["ascend"]["kv_transfer_page_layout"] = key[2]
            for role, tensor in zip(("prefill", "decode"), key[:2], strict=True):
                raw["engine"]["workers"][role]["parallelism"].update(
                    tensor=tensor, attention_data=args.world_size // tensor, moe_expert=args.world_size
                )
            spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
            deployment = spec.backend_deployment
            assert deployment.kv_transfer_provider == "measured"
            models[key] = [
                RustForwardPassPerfModel.best_available(deployment.performance_model_metadata[role]["config"])
                for role in ("prefill", "decode")
            ]
        source, target = models[key]
        predicted = source.estimate_kv_transfer_time_ms(target, row["tokens"])
        assert abs(predicted - row["latency_ms"]) <= 1e-12, (row, predicted)
        results.append(
            {"mapping": list(key), "tokens": row["tokens"], "measured_ms": row["latency_ms"], "queried_ms": predicted}
        )
    result = {
        "scope": "published-table ingestion and canonical query, not independent serving accuracy",
        "rows_checked": len(results),
        "mappings_checked": len(models),
        "metadata": [source.kv_transfer_metadata(target) for source, target in models.values()],
        "rows": results,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows_checked": len(results), "mappings_checked": len(models)}))


if __name__ == "__main__":
    main()
