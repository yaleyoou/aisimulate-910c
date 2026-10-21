import ast
import datetime
import hashlib
import json
import pathlib
import subprocess


def inspect(target, image=False):
    command = ["docker", "image", "inspect", target] if image else ["docker", "inspect", target]
    return json.loads(subprocess.check_output(command, text=True))[0]


def main():
    output = pathlib.Path(__file__).resolve().parent
    container = inspect("glm52-united-v0522")
    image = inspect(container["Image"], image=True)
    allowed = {
        "SERVICE_HOST", "SERVICE_PORT", "COMM_IFNAME", "HCCL_IF_BASE_PORT",
        "CONTEXT_LENGTH", "TP_SIZE", "DP_SIZE", "EP_SIZE", "MEM_FRACTION_STATIC",
        "MAX_RUNNING_REQUESTS", "CHUNKED_PREFILL_SIZE", "MAX_PREFILL_TOKENS",
        "TOKENIZER_WORKER_NUM", "WATCHDOG_TIMEOUT", "ENABLE_DP_ATTENTION",
        "ENABLE_DP_LM_HEAD", "ENABLE_DEEPEP", "DEEPEP_MODE", "ENABLE_DECODE_GRAPH",
        "DECODE_GRAPH_MAX_BS", "DECODE_GRAPH_BATCHES", "SERVED_MODEL_NAME", "SGLANG_NPU_PROFILING",
        "MODEL_LOADER_EXTRA_CONFIG",
        "WEIGHT_LOADER_DISABLE_MMAP",
    }
    environment = dict(item.split("=", 1) for item in container["Config"]["Env"] if item.split("=", 1)[0] in allowed)
    scripts = {}
    for name in ("united-config.sh", "run-united.sh", "run-united-container.sh", "start-united.sh", "start-united-v0522.sh"):
        path = pathlib.Path("/home/youxingye/United deployment/service") / name
        scripts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = {
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "container_name": container["Name"].lstrip("/"),
        "container_id": container["Id"],
        "state": container["State"],
        "image_id": image["Id"],
        "image_digests": image["RepoDigests"],
        "image_architecture": image["Architecture"],
        "image_created": image["Created"],
        "entrypoint": container["Config"]["Entrypoint"],
        "environment": environment,
        "mounted_model_read_only": any(mount["Destination"] == "/models/glm-q" and not mount["RW"] for mount in container["Mounts"]),
        "script_sha256": scripts,
        "canonical_parallelism": {"attention_tp": 8, "attention_dp": 2, "moe_tp": 1, "moe_ep": 16, "pp": 1, "cp": 1, "device_count": 16},
        "workload": {"input_tokens": 8192, "output_tokens": 1024},
        "limitations": ["Container running is not proof of serving readiness.", "CLI environment values need comparison with resolved runtime settings.", "Tiny device probe does not establish collective or full-model correctness."],
    }
    (output / "deployment-baseline.json").write_text(json.dumps(data, indent=2) + "\n")
    logs = subprocess.check_output(["docker", "logs", container["Id"]], stderr=subprocess.STDOUT, text=True)
    (output / "united-v0522-container.log").write_text(logs)
    for line in logs.splitlines():
        if "server_args={" in line:
            resolved = ast.literal_eval(line.split("server_args=", 1)[1])
            fields = {
                "model_path", "dtype", "quantization", "kv_cache_dtype", "context_length",
                "tp_size", "dp_size", "attn_dp_size", "attn_cp_size", "pp_size", "ep_size",
                "moe_dp_size", "mem_fraction_static", "max_running_requests", "max_total_tokens",
                "chunked_prefill_size", "max_prefill_tokens", "page_size", "attention_backend",
                "moe_a2a_backend", "deepep_mode", "enable_dp_lm_head", "speculative_algorithm",
                "cuda_graph_config", "cuda_graph_bs_decode", "cuda_graph_max_bs_decode",
                "cuda_graph_backend_decode", "disable_shared_experts_fusion", "model_impl",
                "model_loader_extra_config",
                "weight_loader_disable_mmap",
            }
            (output / "runtime-resolved-args.json").write_text(json.dumps({key: value for key, value in resolved.items() if key in fields}, indent=2) + "\n")
            break
    print(data["container_name"], data["state"]["Status"], data["image_id"])


if __name__ == "__main__":
    main()
