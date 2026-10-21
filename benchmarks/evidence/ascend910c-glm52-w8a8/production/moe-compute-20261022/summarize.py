import gzip
import hashlib
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    source = root.parent / "real-moe-state-20261022/gmm1-invocations.json.gz"
    with gzip.open(source, "rt") as stream:
        originals = {case["case_id"]: case for case in json.load(stream)}
    planned = json.loads((root / "cases.json").read_text())
    outputs = {}
    inputs = {}
    for filename in ("moe-smoke-result.json", "moe-full-result.json", "moe-reverse-result.json"):
        path = root / filename
        data = json.loads(path.read_text())
        assert data["status"] == "completed", data.get("error")
        assert data["canonical_fpm_data"] is False
        plan_path = root / ("smoke-cases.json" if "smoke" in filename else "cases.json")
        assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == data["case_sha256"]
        expected = json.loads(plan_path.read_text())
        assert [case["case"]["case_id"] for case in data["cases"]] == [case["case_id"] for case in expected]
        assert data["collector_sha256"] == hashlib.sha256((root / "ascend_moe_compute.py").read_bytes()).hexdigest()
        counts = []
        failures = []
        for entry in data["cases"]:
            case = entry["case"]
            original = originals[case["case_id"]]
            for field in original:
                if field != "observations":
                    assert case[field] == original[field], (case["case_id"], field)
            assert case["observation_count"] == len(original["observations"])
            assert case["observations"] == original["observations"][:1]
            assert entry["correctness"]["passed"]
            assert len(entry["correctness"]["checks"]) == entry["correctness"]["sampled_rows"] * 2
            expected_keys = {(op, copies) for op in ("gmm1", "activation", "gmm2", "chain")
                             for copies in data["protocol"]["weight_copies"]}
            observed_keys = [(row["operation"], row["weight_copies"]) for row in entry["measurements"]]
            assert len(observed_keys) == len(expected_keys) and set(observed_keys) == expected_keys
            for row in entry["measurements"]:
                for field in ("samples_ms", "wall_samples_ms"):
                    assert len(row[field]) == data["protocol"]["repeats"]
                    assert all(math.isfinite(value) and value > 0 for value in row[field])
                if row["cv"] > 0.1:
                    failures.append({"case_id": case["case_id"], "operation": row["operation"],
                                     "weight_copies": row["weight_copies"], "cv": row["cv"]})
            counts.append(len(entry["measurements"]))
        outputs[filename] = {"cases": len(counts), "measurements": sum(counts), "cv_failures": failures}
        inputs[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    full = json.loads((root / "moe-full-result.json").read_text())
    reverse = json.loads((root / "moe-reverse-result.json").read_text())
    comparisons = []
    for first, second in zip(full["cases"], reverse["cases"]):
        first_grid = {(row["operation"], row["weight_copies"]): row for row in first["measurements"]}
        second_grid = {(row["operation"], row["weight_copies"]): row for row in second["measurements"]}
        comparisons.append({
            "case_id": first["case"]["case_id"],
            "logical_rows": sum(first["case"]["expert_counts"]),
            "storage_rows": first["case"]["input_shape"][0],
            "eight_to_sixteen_copy_ratio": {
                op: second_grid[op, 16]["median_ms"] / second_grid[op, 8]["median_ms"]
                for op in ("gmm1", "gmm2", "chain")},
            "chain_repeat_ratio_changed_graph_calls": second_grid["chain", 8]["median_ms"]
                / first_grid["chain", 8]["median_ms"],
            "sum_components_over_chain": sum(second_grid[op, 8]["median_ms"]
                for op in ("gmm1", "activation", "gmm2")) / second_grid["chain", 8]["median_ms"],
        })
    report = {
        "schema": "aisimulate-ascend-moe-protocol-validation/v1",
        "status": "complete_experimental_protocol_comparison",
        "production_prediction_ready": False,
        "source_state_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_sha256": inputs,
        "selected_observed_cases": len(planned),
        "outputs": outputs,
        "comparisons": comparisons,
        "limitations": [
            "Selected actual count vectors; seeded values, not actual checkpoint weights or activations.",
            "Zero-count cases validate execution only; there are no populated output rows to check numerically.",
            "Standalone graph timing also used for eager-source geometry; eager launch cost remains unqualified.",
            "8-to-16-copy convergence only covers these cases. The 16-versus32-call repeat changes timing scope.",
            "CV failures remain in raw data and are not accepted timing points.",
            "No claim of complete operator coverage, canonical forward accuracy, PD or search readiness.",
        ],
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "outputs": outputs}))


if __name__ == "__main__":
    main()
