"""Exercise AscendTPDispatcher's global routing with expert-partitioned weights."""

import importlib
import json
import traceback
from types import SimpleNamespace

import torch
import torch_npu

importlib.import_module("sglang.srt.layers.quantization")
from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW8A8Int8MoEMethod
from sglang.srt.hardware_backend.npu.utils import init_npu_backend, npu_format_cast
from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher

init_npu_backend()
torch.npu.set_device(0)
torch.set_grad_enabled(False)
dispatcher = AscendTPDispatcher(SimpleNamespace(num_experts=256, top_k=8))
dispatcher.set_quant_config({"dispatcher_output_dtype": "int8"})
inputs = torch.randn((8, 6144), device="npu", dtype=torch.bfloat16)
ids = torch.arange(248, 256, device="npu", dtype=torch.int32).repeat(8, 1)
probabilities = torch.full((8, 8), 0.125, device="npu", dtype=torch.float32)
result = {
    "scope": "Isolated actual dispatcher/GMM invocation with EP2 rank0 metadata, not a full distributed forward",
    "global_experts": 256, "local_experts": 128, "selected_global_experts": list(range(248, 256)),
    "framework_commit": "baa9befa381511cd4d56f83980951e5a45fffc6d",
    "torch_npu": torch_npu.__version__,
}
try:
    routed = dispatcher.dispatch(inputs, (probabilities, ids, None))
    weights = SimpleNamespace(
        w13_weight=npu_format_cast(torch.ones((128, 6144, 512), device="npu", dtype=torch.int8)),
        w13_weight_scale=torch.full((128, 512), 0.125, device="npu", dtype=torch.bfloat16),
    )
    result["dispatch_count_shape"] = list(routed.expert_tokens.shape)
    result["nonlocal_routed_tokens"] = int(routed.expert_tokens[128:].sum().cpu())
    output = NPUW8A8Int8MoEMethod().apply(
        weights, routed.hidden_states, routed.expert_tokens, routed.hidden_states_scale, torch.bfloat16, "w13", 1
    )
    torch.npu.synchronize()
    result.update(status="executed", output_shape=list(output.shape))
except Exception:
    result.update(status="failed", error=traceback.format_exc())
print(json.dumps(result, indent=2), flush=True)
