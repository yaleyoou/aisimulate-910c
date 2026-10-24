import importlib
import json

import torch
import torch_npu

importlib.import_module("sglang.srt.layers.quantization")
from sglang.srt.hardware_backend.npu.moe.init_routing import NPUMoEInitRouting_v2
from sglang.srt.hardware_backend.npu.utils import init_npu_backend

init_npu_backend()
torch.npu.set_device(0)
torch.manual_seed(910)
inputs = (torch.randint(-31, 32, (8, 6144), device="npu").float() / 32).bfloat16()
ids = (torch.arange(64, device="npu", dtype=torch.int32) % 256).reshape(8, 8)
routed, mapping, counts, scales = NPUMoEInitRouting_v2(1)._init_routing(inputs, ids, 256, 8)
torch.npu.synchronize()
source = inputs.float().cpu()
expected_scales = source.abs().amax(-1) / 127
expected = (source / expected_scales[:, None]).round().clamp(-128, 127)
observed = routed.float().cpu()
differences = (observed[:, None, :32] - expected[None, :, :32]).abs().amax(-1)
print(json.dumps({
    "torch_npu": torch_npu.__version__,
    "routed_shape": list(routed.shape),
    "mapping": mapping.cpu().tolist(),
    "nonzero_counts": counts[counts > 0].cpu().tolist(),
    "source_token_for_row": differences.argmin(-1).tolist(),
    "best_error_per_row": differences.amin(-1).tolist(),
    "scales": scales.cpu().tolist(),
    "expected_scales": expected_scales.tolist(),
}, indent=2))
