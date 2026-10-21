import json
import pathlib
import re


def main():
    root = pathlib.Path(__file__).resolve().parent
    info = json.loads((root / "serving-smoke/server-info.json").read_text())
    logs = (root / "united-v0522-container.log").read_text()
    cache_pattern = re.compile(r"\[([^\]]+) DP(\d+) TP(\d+) EP(\d+)\] KV Cache is allocated\. dtype: ([^,]+), #tokens: (\d+), KV size: ([\d.]+) GB")
    graph_pattern = re.compile(r"\[([^\]]+) DP(\d+) TP(\d+) EP(\d+)\] Capture target decode NPU graph end\. bs=(\[[^\]]+\]), elapsed=([\d.]+) s, mem usage=([\d.]+) GB, avail mem=([\d.]+) GB")
    cache_rows = [{"timestamp_container_utc": match[0], "attention_dp_rank": int(match[1]), "tp_rank": int(match[2]), "ep_rank": int(match[3]), "dtype": match[4], "capacity_tokens": int(match[5]), "size_log_gb_rounded": float(match[6])} for match in cache_pattern.findall(logs)]
    graph_rows = [{"timestamp_container_utc": match[0], "attention_dp_rank": int(match[1]), "tp_rank": int(match[2]), "ep_rank": int(match[3]), "batch_sizes": json.loads(match[4]), "capture_seconds": float(match[5]), "memory_log_gb_rounded": float(match[6]), "available_log_gb_rounded": float(match[7])} for match in graph_pattern.findall(logs)]
    scheduler_lines = [line for line in logs.splitlines() if "max_total_num_tokens=" in line and "max_running_requests=" in line]
    assert len(cache_rows) == 16, f"Expected 16 cache observations, got {len(cache_rows)}"
    assert len(graph_rows) == 16, f"Expected 16 graph completions, got {len(graph_rows)}"
    assert all(row["capacity_tokens"] == info["max_total_num_tokens"] for row in cache_rows)
    data = {
        "status": info["status"], "version": info["version"],
        "max_total_num_tokens_per_attention_dp": info["max_total_num_tokens"],
        "max_req_input_len_reported": info["max_req_input_len"],
        "startup_time": info["startup_time"], "runtime_random_seed": info["random_seed"],
        "cache_observations": cache_rows, "graph_capture_observations": graph_rows,
        "scheduler_log_lines": scheduler_lines,
        "source_files": ["serving-smoke/server-info.json", "united-v0522-container.log"],
        "limits": ["Rounded GB log values are not exact allocation byte measurements.", "TP ranks replicate a group's cache capacity; do not multiply token capacity by 16.", "This is investigation evidence, not an imported or finalized FPM memory profile.", "Configured context length180000 does not override the observed capacity/input limits."],
    }
    (root / "runtime-memory-observations.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Validated 16 cache observations and 16 completed graph captures")


if __name__ == "__main__":
    main()
