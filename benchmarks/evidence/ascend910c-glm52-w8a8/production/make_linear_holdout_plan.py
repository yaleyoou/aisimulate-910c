"""Generate independent intermediate-M cases through the same declared geometry."""

import argparse
import json
from pathlib import Path
import shutil

import yaml
from collector.sglang.ascend_linear_cases import get_ascend_linear_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = Path(__file__).resolve().parents[4] / "python/aisimulate/collector/cases"
    (args.output_dir / "base_ops").mkdir()
    (args.output_dir / "models").mkdir()
    model_name = "GlmMoeDsaForCausalLM_cases.yaml"
    shutil.copyfile(cases / "models" / model_name, args.output_dir / "models" / model_name)
    declaration = yaml.safe_load((cases / "base_ops/ascend_linear.yaml").read_text())
    values = declaration["common_case_values"]["ascend_linear"]
    values["token_counts"] = [3, 12, 48, 96, 3072]
    values["request_counts"] = [3, 12, 48]
    (args.output_dir / "base_ops/ascend_linear.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    plan = get_ascend_linear_cases(args.model, args.output_dir)
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    policy = {
        "purpose": "Independent held-out token interpolation; these measurements must not be inserted into the tested table.",
        "quality": "Exclude observations with CV>10%; report every exclusion. Query exact source/layout/precision in native engine.",
        "acceptance": "Report median,p90,maximum absolute relative error by execution/quantization; 10% is the per-case comparison target, not whole-forward qualification.",
    }
    (args.output_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(json.dumps({"cases": len(plan["cases"]), "policy": policy}))


if __name__ == "__main__":
    main()
