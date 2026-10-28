import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as parquet
import yaml
from collector.sglang.ascend_curve_append import physical_coordinate, physical_curve


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("attention", "bmm", "vector", "linear"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("campaigns", type=Path, nargs="+")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[4]
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"], text=True
    ).strip()
    family = {"attention": "attention", "bmm": "mla_bmm", "vector": "vector", "linear": "gemm"}[args.operation]
    live = repository / f"python/aisimulate/src/aisimulate_core/systems/data/ascend910c/{family}/sglang/0.5.22"
    table_name = f"ascend_{args.operation}_perf"
    filename = table_name + ".parquet"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    before = args.output_dir / "before"
    before.mkdir()
    for source, name in (
        (live / filename, filename),
        (live / "collection_meta.yaml", "collection_meta.yaml"),
        (live / "metadata/provenance.json", "provenance.json"),
    ):
        shutil.copyfile(source, before / name)
    exports = []
    for campaign in args.campaigns:
        status = json.loads((campaign / "campaign.json").read_text())
        if status["status"] != "completed" or not all(job["expected_outcome"] for job in status["jobs"]):
            raise ValueError(f"Incomplete campaign: {campaign}")
        observations = sorted(campaign.glob("shard-*.json"))
        record = json.loads(observations[0].read_text())
        runtime = record["runtime"]
        identity = {
            key: value
            for key, value in runtime.items()
            if key not in ("device", "device_index", "cpu_affinity", "serving_setup", "host")
        }
        identity.update(
            hardware=runtime["device"],
            collector_source_sha256=record["collector_source_sha256"],
            image_digest="quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443",
        )
        exporter = importlib.import_module(f"collector.sglang.ascend_{args.operation}_export")
        rows, report = exporter.validated_rows(campaign / "plan.json", observations, identity)
        if report.get("unqualified"):
            raise ValueError(f"Unqualified points require review: {campaign}: {len(report['unqualified'])}")
        identity_path = campaign / "runtime-identity.json"
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
        destination = campaign / "export"
        subprocess.run(
            [
                sys.executable,
                "-m",
                f"collector.sglang.ascend_{args.operation}_export",
                *map(str, observations),
                "--plan",
                str(campaign / "plan.json"),
                "--runtime-identity",
                str(identity_path),
                "--output-dir",
                str(destination),
                "--collector-ref",
                revision,
            ],
            check=True,
        )
        exports.append(destination)
    merged = args.output_dir / "merged"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "collector.sglang.ascend_curve_append",
            "--operation",
            args.operation,
            "--extend-points",
            "--output-dir",
            str(merged),
            str(before),
            *map(str, exports),
        ],
        check=True,
    )
    original = parquet.read_table(before / filename).to_pylist()
    combined = parquet.read_table(merged / filename).to_pylist()
    key = lambda row: (physical_curve(args.operation, row), physical_coordinate(args.operation, row))
    indexed = {key(row): row for row in combined}
    if len(indexed) != len(combined):
        raise ValueError("Duplicate physical coordinates")
    for row in original:
        retained = indexed[key(row)]
        if any(value != retained[field] for field, value in row.items() if field != "runtime_identity_sha256"):
            raise ValueError("Historical observation changed")
    old_metadata = yaml.safe_load((before / "collection_meta.yaml").read_text())
    metadata = yaml.safe_load((merged / "collection_meta.yaml").read_text())
    metadata["tables"] = {**old_metadata["tables"], **metadata["tables"]}
    staged = live / (filename + ".new")
    shutil.copyfile(merged / filename, staged)
    staged.replace(live / filename)
    shutil.copyfile(merged / "provenance.json", live / "metadata/provenance.json")
    (live / "collection_meta.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    publication = {
        "operation": args.operation,
        "source_revision": revision,
        "old_rows_preserved": len(original),
        "new_rows": len(combined) - len(original),
        "total_rows": len(combined),
        "old_observations_unchanged_except_composite_identity": True,
        "table_sha256": sha256(live / filename),
        "campaigns": [str(path) for path in args.campaigns],
    }
    (args.output_dir / "publication.json").write_text(json.dumps(publication, indent=2) + "\n")
    print(json.dumps(publication))


if __name__ == "__main__":
    main()
