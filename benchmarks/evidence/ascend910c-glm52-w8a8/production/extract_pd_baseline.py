import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
}
DEPLOYMENTS = {1: "colocated_two_nodes", 2: "1p1d", 3: "2p1d", 4: "3p1d", 5: "1p3d", 6: "2p2d"}
WORKLOADS = {1: (8192, 1024), 2: (32768, 1024), 3: (1024, 32768), 4: (32768, 300)}
COLUMNS = (
    "concurrency",
    "completed_requests",
    "input_tokens_per_s",
    "output_tokens_per_s",
    "mean_ttft_s",
    "p99_ttft_s",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "reported_output_tokens_per_s_per_device",
    "reported_cache_hit_ratio",
)


def sheet_rows(content):
    with ZipFile(io.BytesIO(content)) as workbook:
        strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            strings = [
                "".join(item.itertext()) for item in ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
            ]
        sheets = sorted(
            name for name in workbook.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if len(sheets) != 1:
            raise ValueError("Expected one worksheet per embedded workbook")
        rows = []
        for row in ElementTree.fromstring(workbook.read(sheets[0])).findall(".//s:row", NAMESPACES):
            values = {}
            for cell in row.findall("s:c", NAMESPACES):
                value = cell.find("s:v", NAMESPACES)
                text = "" if value is None else value.text or ""
                if cell.attrib.get("t") == "s":
                    text = strings[int(text)]
                elif cell.attrib.get("t") == "inlineStr":
                    text = "".join(cell.find("s:is", NAMESPACES).itertext())
                values["".join(character for character in cell.attrib["r"] if character.isalpha())] = text
            rows.append(values)
        return rows


def extract(source):
    records, workbooks, paragraphs = [], [], []
    with ZipFile(source) as document:
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in ElementTree.fromstring(document.read("word/_rels/document.xml.rels"))
        }
        title = ""
        for block in ElementTree.fromstring(document.read("word/document.xml")).find("w:body", NAMESPACES):
            text = "".join(element.text or "" for element in block.findall(".//w:t", NAMESPACES))
            if text:
                paragraphs.append(text)
            if text.startswith("2."):
                title = text
            for element in block.iter():
                if not element.tag.endswith("}OLEObject"):
                    continue
                target = targets[element.attrib["{" + NAMESPACES["r"] + "}id"]]
                content = document.read("word/" + target)
                rows = sheet_rows(content)
                section = title.split()[0].split(".")
                deployment = DEPLOYMENTS[int(section[1])]
                workload = int(section[2])
                input_tokens, output_tokens = WORKLOADS[workload]
                if rows[0].get("E") != "Mean TTFT (s)" or rows[0].get("G") != "Mean TPOT (ms)":
                    raise ValueError("Unexpected metric units or definitions")
                workbooks.append(
                    {
                        "section": title,
                        "path": target,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "headers": rows[0],
                    }
                )
                for row in rows[1:]:
                    if not row.get("A"):
                        continue
                    record = {
                        "deployment": deployment,
                        "workload": f"w{workload}",
                        "section": title,
                        "source_workbook": target,
                        "input_tokens_assumed_from_label": input_tokens,
                        "output_tokens_assumed_from_label": output_tokens,
                    }
                    for index, column in enumerate(COLUMNS):
                        value = row.get(chr(ord("A") + index), "")
                        record[column] = float(value) if value else None
                    for column in ("concurrency", "completed_requests"):
                        record[column] = int(record[column])
                    record["estimated_decode_phase_occupancy"] = (
                        record["output_tokens_per_s"]
                        * record["mean_tpot_ms"]
                        / 1000
                        * (output_tokens - 1)
                        / output_tokens
                    )
                    record["estimated_inflight_occupancy"] = (
                        record["output_tokens_per_s"]
                        / output_tokens
                        * (record["mean_ttft_s"] + (output_tokens - 1) * record["mean_tpot_ms"] / 1000)
                    )
                    records.append(record)
    return {
        "source": {"path": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "notes": [
            "Metrics are extracted from embedded Excel XML, not OCR or the document's summary prose.",
            "Excel headers identify mean and p99; prose sometimes calls related values p50. "
            "Preserve spreadsheet definitions.",
            "Token lengths are inferred from workload labels; raw request/tokenizer records are unavailable.",
            "Occupancy is an arithmetic diagnostic assuming fixed output lengths "
            "and per-request TPOT=(last-first)/(tokens-1).",
            "Decode-phase occupancy includes host/streaming delays and finite-run fill/drain; "
            "it is not a measured per-rank running batch.",
            "The document does not identify immutable serving image/CANN/HCCL versions, "
            "arrival timestamps, KV capacity, or per-rank traces.",
            "Cache-workload labels and claimed hit percentages do not define a reproducible cache trace.",
        ],
        "workbooks": workbooks,
        "paragraphs": paragraphs,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.docx)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "baseline.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["records"][0]))
        writer.writeheader()
        writer.writerows(result["records"])
    print(
        json.dumps(
            {
                "workbooks": len(result["workbooks"]),
                "records": len(result["records"]),
                "source_sha256": result["source"]["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
