"""Select complete compatible PD curves without mixing their timing protocols."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def select_curves(campaigns):
    if not campaigns:
        raise ValueError("At least one verified PD campaign is required")
    runtime = campaigns[0]["metadata"]["runtime"]
    setup = campaigns[0]["report"]["identity"]["serving_setup"]
    candidates = defaultdict(list)
    for index, campaign in enumerate(campaigns):
        if campaign["metadata"]["runtime"] != runtime or campaign["report"]["identity"]["serving_setup"] != setup:
            raise ValueError("Cannot merge PD measurements from different runtime or serving conditions")
        identity = campaign["report"]["identity"]
        expected_identity = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        groups = defaultdict(list)
        for row in campaign["rows"]:
            if row["runtime_identity_sha256"] != expected_identity:
                raise ValueError("PD table and campaign identity differ")
            groups[row["source_tp"], row["destination_tp"], row["page_layout"]].append(row)
        for key, rows in groups.items():
            lengths = [row["tokens"] for row in rows]
            if len(lengths) != len(set(lengths)):
                raise ValueError("Duplicate physical PD point")
            quality = (min(lengths) <= 8192 <= max(lengths), max(lengths) - min(lengths), len(lengths), -index)
            candidates[key].append((quality, index, rows))
    selected, decisions, counts = [], [], defaultdict(int)
    for key, alternatives in sorted(candidates.items()):
        _, index, rows = max(alternatives, key=lambda item: item[0])
        selected.extend(sorted(rows, key=lambda row: row["tokens"]))
        counts[index] += len(rows)
        decisions.append(
            {
                "source_tp": key[0],
                "destination_tp": key[1],
                "page_layout": key[2],
                "campaign": index,
                "measured_tokens": sorted(row["tokens"] for row in rows),
                "collection_identity_sha256": rows[0]["runtime_identity_sha256"],
            }
        )
    return selected, decisions, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    import yaml
    from collector.provenance import append_collection_event, write_collection_meta

    campaigns = []
    for directory in args.campaign:
        report_path = directory / "provenance.json"
        if not report_path.is_file():
            report_path = directory / "metadata/provenance.json"
        table = directory / "ascend_pd_perf.parquet"
        campaigns.append(
            {
                "report": json.loads(report_path.read_text()),
                "metadata": yaml.safe_load((directory / "collection_meta.yaml").read_text()),
                "rows": parquet.read_table(table).to_pylist(),
                "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
            }
        )
    rows, decisions, counts = select_curves(campaigns)
    entry = None
    count = 0
    for index, contributed in sorted(counts.items()):
        source = campaigns[index]["metadata"]
        event = dict(source["tables"]["ascend_pd_perf"])
        event.update(
            rows=contributed,
            source_campaign_rows=event["rows"],
            source_campaign_status=event["status"],
            runtime=source["runtime"],
        )
        count += contributed
        entry = (
            event if entry is None else append_collection_event(entry, event, table="ascend_pd_perf", merged_rows=count)
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parquet.write_table(arrow.Table.from_pylist(rows), args.output_dir / "ascend_pd_perf.parquet", compression="zstd")
    write_collection_meta(
        args.output_dir, campaigns[0]["metadata"]["runtime"], {"ascend_pd_perf": entry}, provenance_tier="collected"
    )
    report = {
        "selection": (
            "whole curves: cover8192 first, largest measured span, most points, then earlier input; "
            "latency never selects a winner"
        ),
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
