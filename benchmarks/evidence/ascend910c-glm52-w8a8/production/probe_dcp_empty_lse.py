import json
from types import SimpleNamespace

import torch
import torch_npu
from sglang.srt.hardware_backend.npu.attention.dsa_dcp import forward_dcp_sparse_attention
from sglang.srt.hardware_backend.npu.utils import init_npu_backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode

init_npu_backend()
torch.npu.set_device(0)
query = torch.randn((1, 32, 512), device="npu", dtype=torch.bfloat16)
query_rope = torch.randn((1, 32, 64), device="npu", dtype=torch.bfloat16)
cache = torch.zeros((1, 209, 128, 1, 512), device="npu", dtype=torch.bfloat16)[0]
rope_cache = torch.zeros((1, 209, 128, 1, 64), device="npu", dtype=torch.bfloat16)[0]
partial, lse = forward_dcp_sparse_attention(
    q_nope=query, q_rope=query_rope, k_nope=cache, k_rope=rope_cache,
    topk_indices=torch.full((1, 1, 2048), -1, device="npu", dtype=torch.int32),
    actual_seq_lengths_query=torch.tensor([1], device="npu", dtype=torch.int32),
    forward_metadata=SimpleNamespace(
        dcp_block_tables=torch.tensor([[1]], device="npu", dtype=torch.int32),
        dcp_seq_lens=torch.tensor([4], device="npu", dtype=torch.int32),
    ),
    forward_batch=SimpleNamespace(forward_mode=ForwardMode.EXTEND),
    speculative_num_draft_tokens=None, scaling=1 / 16,
)
torch.npu.synchronize()
print(json.dumps({
    "torch_npu": torch_npu.__version__, "scope": "All sparse indices unowned with a nonempty local physical KV stripe",
    "lse": lse.cpu().tolist(), "lse_exp": lse.exp().cpu().tolist(),
    "partial_finite": bool(partial.isfinite().all().cpu()), "partial_abs_max": float(partial.abs().max().cpu()),
}, indent=2))
