"""Record installed serving source identity without redistributing vendor code."""

import hashlib
import json
import subprocess
from pathlib import Path

CONTAINER = "glm52-united-v0522"
SOURCE_ROOT = "/sgl-workspace/sglang"
FILES = (
    "srt/models/glm4_moe.py",
    "srt/models/deepseek_v2.py",
    "srt/configs/model_config.py",
    "srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py",
    "srt/hardware_backend/npu/attention/ascend_backend.py",
    "srt/hardware_backend/npu/quantization/linear_method_npu.py",
    "srt/hardware_backend/npu/quantization/moe_methods.py",
    "srt/hardware_backend/npu/moe/matmul.py",
    "srt/hardware_backend/npu/moe/activation.py",
    "srt/layers/attention/dsa/dsa_npu_indexer.py",
    "srt/layers/moe/moe_runner/ascend.py",
    "srt/layers/moe/token_dispatcher/deepep.py",
    "srt/managers/scheduler_components/profiler_manager.py",
    "srt/managers/io_struct.py",
    "srt/entrypoints/http_server.py",
)


def command(*arguments):
    return subprocess.check_output(["docker", "exec", CONTAINER, *arguments])


def main():
    revision = command("git", "-C", SOURCE_ROOT, "rev-parse", "HEAD").decode().strip()
    inventory = {
        "repository": "https://github.com/sgl-project/sglang",
        "commit": revision,
        "container": CONTAINER,
        "files": [],
        "scope": "Hashes and source locations only; upstream implementations are invoked, not copied.",
    }
    for relative in FILES:
        path = "python/sglang/" + relative
        installed = command("cat", SOURCE_ROOT + "/" + path)
        committed = command("git", "-C", SOURCE_ROOT, "show", revision + ":" + path)
        inventory["files"].append(
            {
                "path": path,
                "sha256": hashlib.sha256(installed).hexdigest(),
                "matches_commit": installed == committed,
                "lines": len(installed.splitlines()),
            }
        )
    if not all(item["matches_commit"] for item in inventory["files"]):
        raise RuntimeError("Installed serving implementations differ from pinned commit")
    Path(__file__).with_name("serving-source-inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
