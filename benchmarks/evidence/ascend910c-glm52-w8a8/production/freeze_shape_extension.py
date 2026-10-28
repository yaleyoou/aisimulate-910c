import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import pyarrow.parquet as parquet


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("roots", type=Path, nargs="+")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[4]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    members = set()
    for root in args.roots:
        root = root.resolve(strict=True)
        if root.is_dir():
            members.update(path for path in root.rglob("*") if path.is_file())
        else:
            members.add(root)
    archive = args.output_dir / "raw-evidence.tar.gz"
    inventory = []
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(members):
            relative = str(path.relative_to(repository))
            bundle.add(path, arcname=relative, recursive=False)
            inventory.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    tables = []
    system_data = repository / "python/aisimulate/src/aisimulate_core/systems/data/ascend910c"
    for path in sorted(system_data.rglob("*.parquet")):
        rows = parquet.read_table(path).to_pylist()
        dimensions = {}
        for field in (
            "operation",
            "role",
            "phase",
            "execution_mode",
            "quant_mode",
            "local_heads",
            "local_experts",
            "num_ranks",
            "ranks_per_node",
            "distribution",
            "tokens",
            "m",
            "received_tokens",
            "active_batch",
            "padded_batch",
            "sequence_length",
            "prefix",
            "message_bytes",
        ):
            values = {row[field] for row in rows if row.get(field) is not None}
            if values:
                dimensions[field] = sorted(values)
        tables.append(
            {"path": str(path.relative_to(system_data)), "rows": len(rows), "sha256": digest(path), "axes": dimensions}
        )
    quality = []
    for path in sorted(members):
        if path.name == "quality-report.json":
            report = json.loads(path.read_text())
            quality.append(
                {
                    "campaign": path.parent.name,
                    **{key: report[key] for key in ("attempted", "qualified", "unqualified")},
                }
            )
    manifest = {
        "scope": "Frozen measured observations, controls, quality decisions and publication evidence; "
        "not serving accuracy acceptance.",
        "archive": archive.name,
        "archive_sha256": digest(archive),
        "members": inventory,
        "quality": quality,
        "tables": tables,
        "per_family_rows": {table["path"]: table["rows"] for table in tables},
        "limitations": [
            "Axis values are marginal coverage, not a complete Cartesian product; "
            "inspect the table's full curve key before assuming a shape is measured.",
            "Rejected original measurements remain in the archive; only their missing coordinates "
            "are supplied by independent healing campaigns.",
            "Native exact-point checks verify integration, not interpolation, extrapolation or whole-forward accuracy.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"archive": str(archive), "files": len(inventory), "bytes": archive.stat().st_size, "tables": len(tables)}
        )
    )


if __name__ == "__main__":
    main()
