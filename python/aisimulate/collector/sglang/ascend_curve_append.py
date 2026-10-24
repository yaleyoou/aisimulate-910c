"""Append disjoint measured linear/vector curves without replacing existing points."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def physical_curve(operation, row):
    if operation == "linear":
        return tuple(row[field] for field in ("quant_mode", "n", "k", "weight_format", "quant_bias", "execution_mode"))
    if operation == "vector":
        return row["role"], tuple(json.loads(row["dimensions"])), row["execution_mode"]
    raise ValueError("Only linear and vector curve extension is supported")


def collection_identities(report):
    if "identity" in report:
        identity = report["identity"]
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {digest: identity}
    identities = {}
    for source in report["campaigns"]:
        identities.update(collection_identities(source["report"]))
    return identities


def append_curves(operation, campaigns):
    if not campaigns:
        raise ValueError("At least one verified campaign is required")
    runtime = campaigns[0]["metadata"]["runtime"]
    rows, decisions, owners, counts = [], [], set(), Counter()
    setup = None
    axis = "m" if operation == "linear" else "tokens"
    for index, campaign in enumerate(campaigns):
        if campaign["metadata"]["runtime"] != runtime:
            raise ValueError("Different runtime conditions cannot share a table")
        identities = collection_identities(campaign["report"])
        for identity in identities.values():
            current_setup = identity["serving_setup"]
            if setup is not None and setup != current_setup:
                raise ValueError("Different serving setups cannot share a table")
            setup = current_setup
        groups = defaultdict(list)
        for row in campaign["rows"]:
            if row["runtime_identity_sha256"] not in identities:
                raise ValueError("Table row has no verified collection identity")
            groups[physical_curve(operation, row)].append(row)
        for key, values in sorted(groups.items()):
            if key in owners:
                raise ValueError(
                    "Overlapping curves require a separate coverage decision; point replacement is forbidden"
                )
            owners.add(key)
            samples = [row[axis] for row in values]
            identities_in_curve = {row["runtime_identity_sha256"] for row in values}
            if len(samples) != len(set(samples)) or len(identities_in_curve) != 1:
                raise ValueError("A curve cannot contain duplicate points or mixed collection protocols")
            rows.extend(sorted(values, key=lambda row: row[axis]))
            counts[index] += len(values)
            decisions.append(
                {
                    "key": key,
                    "campaign": index,
                    "points": sorted(samples),
                    "collection_identity_sha256": next(iter(identities_in_curve)),
                }
            )
    return rows, decisions, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("linear", "vector"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("campaign", type=Path, nargs="+")
    args = parser.parse_args()
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    import yaml
    from collector.provenance import append_collection_event, write_collection_meta

    name = "ascend_" + args.operation + "_perf"
    campaigns = []
    for directory in args.campaign:
        provenance = directory / "provenance.json"
        if not provenance.is_file():
            provenance = directory / "metadata/provenance.json"
        table = directory / (name + ".parquet")
        campaigns.append(
            {
                "report": json.loads(provenance.read_text()),
                "metadata": yaml.safe_load((directory / "collection_meta.yaml").read_text()),
                "rows": parquet.read_table(table).to_pylist(),
                "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
            }
        )
    rows, decisions, counts = append_curves(args.operation, campaigns)
    entry, total = None, 0
    for index, contributed in sorted(counts.items()):
        metadata = campaigns[index]["metadata"]
        event = dict(metadata["tables"][name])
        event.update(
            rows=contributed,
            source_campaign_rows=event["rows"],
            source_campaign_status=event["status"],
            runtime=metadata["runtime"],
        )
        total += contributed
        entry = event if entry is None else append_collection_event(entry, event, table=name, merged_rows=total)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parquet.write_table(arrow.Table.from_pylist(rows), args.output_dir / (name + ".parquet"), compression="zstd")
    write_collection_meta(
        args.output_dir, campaigns[0]["metadata"]["runtime"], {name: entry}, provenance_tier="collected"
    )
    report = {
        "selection": "Append disjoint whole curves; existing curves are retained unchanged. Overlaps are rejected.",
        "curves": decisions,
        "campaigns": [
            {"table_sha256": campaign["table_sha256"], "report": campaign["report"]} for campaign in campaigns
        ],
        "rows": len(rows),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "curves": len(decisions)}))


if __name__ == "__main__":
    main()
