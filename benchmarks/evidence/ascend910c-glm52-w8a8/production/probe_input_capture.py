"""Check observer transparency through the actual NPU custom-op graph path."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys


setup = json.loads(Path("/workspace/benchmarks/evidence/ascend910c-glm52-w8a8/production/serving-setup.json").read_text())
for name, value in setup["environment"].items():
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
os.sched_setaffinity(0, setup["worker_cpu_affinity"]["0"])
import torch
import torch_npu
from sglang.srt.hardware_backend.npu.utils import init_npu_backend

sys.path.insert(0, "/workspace/python/aisimulate")
from collector.sglang.ascend_input_capture import InputCapture, ObservedCallable

torch.npu.set_device(0)
init_npu_backend()
torch.manual_seed(910)
state = SimpleNamespace(thread=SimpleNamespace(capturing_graph=1, recording_forward=False))
observer = InputCapture(state, torch, torch_npu)
original = torch.ops.npu.batch_matmul_transpose


def invoke(*args, **kwargs):
    result = original(*args, **kwargs)
    observer.record("npu.batch_matmul_transpose", {"positional": args, **kwargs}, result)
    return result


torch.ops.npu.batch_matmul_transpose = ObservedCallable(original, invoke)
inputs = torch.randn(8, 8, 512, device="npu", dtype=torch.bfloat16)
weights = torch.randn(8, 512, 256, device="npu", dtype=torch.bfloat16) / 512**0.5
output = torch.empty(8, 8, 256, device="npu", dtype=torch.bfloat16)
for _ in range(3):
    torch.ops.npu.batch_matmul_transpose(inputs, weights, output)
torch.npu.synchronize()
observer.graphs[1] = []
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    torch.ops.npu.batch_matmul_transpose(inputs, weights, output)
for _ in range(2):
    inputs.neg_()
    graph.replay()
    torch.npu.synchronize()
    expected = torch.bmm(inputs.float().cpu().transpose(0, 1), weights.float().cpu()).transpose(0, 1)
    torch.testing.assert_close(output.float().cpu(), expected, rtol=0.012, atol=0.02)
record = {"status": "passed", "checks": 2, "calls": observer.snapshot(1),
          "default_overload_preserved": torch.ops.npu.batch_matmul_transpose.default is original.default}
Path("/campaign/input-capture-probe.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"status": "passed", "calls": len(record["calls"]), "checks": 2}))
