"""Inspect BF16 kernel dispatch at interpolation failures; not timing publications."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    setup = json.loads(args.setup.read_text())
    for name, value in setup["environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    os.sched_setaffinity(0, setup["worker_cpu_affinity"]["0"])
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.set_num_threads(8)
    torch.manual_seed(910)
    manifest = {
        "scope": "Profiled implementation/tiling diagnostics; never use profiler durations as unprofiled latency rows",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": [],
    }
    for tokens, width in ((40, 32), (48, 32), (56, 32), (80, 128), (96, 128), (112, 128),
                          (2816, 128), (3072, 128), (3328, 128)):
        inputs = torch.randn(tokens, 6144, device="npu", dtype=torch.bfloat16)
        raw = torch.randn(width, 6144, device="npu", dtype=torch.bfloat16) / 6144**0.5
        weights = [raw.clone() for _ in range(64)]

        def batch():
            return [torch.nn.functional.linear(inputs, weights[index % 64]) for index in range(128)]

        for _ in range(10):
            batch()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = batch()
        for _ in range(10):
            graph.replay()
        inputs.neg_()
        graph.replay()
        torch.npu.synchronize()
        rows = [0, tokens // 2, tokens - 1]
        expected = inputs[rows].float().cpu() @ raw.float().cpu().transpose(0, 1)
        torch.testing.assert_close(outputs[-1][rows].float().cpu(), expected, rtol=0.012, atol=0.02)
        label = f"m{tokens}-n{width}"
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output / label)),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            ),
        ) as profile:
            graph.replay()
            torch.npu.synchronize()
            profile.step()
        manifest["cases"].append({"m": tokens, "n": width, "k": 6144, "execution_mode": "graph",
                                  "weight_format": torch_npu.get_npu_format(raw), "directory": label})
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        del graph, outputs, weights, raw, inputs
        gc.collect()
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
