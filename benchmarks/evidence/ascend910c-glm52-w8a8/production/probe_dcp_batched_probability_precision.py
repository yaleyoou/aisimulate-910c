"""Probe the two-key prefill shape that triggered partial-attention checks."""

import json
from types import SimpleNamespace

import torch
import torch_npu
from sglang.srt.hardware_backend.npu.attention.dsa_dcp import forward_dcp_sparse_attention
from sglang.srt.hardware_backend.npu.utils import init_npu_backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode

init_npu_backend()
torch.npu.set_device(0)
query = torch.zeros((16, 32, 512), device="npu", dtype=torch.bfloat16)
query[:, :, 0] = 1
query_rope = torch.zeros((16, 32, 64), device="npu", dtype=torch.bfloat16)
cache_cpu = torch.zeros((1, 209, 128, 1, 512), dtype=torch.bfloat16)
for page in range(1, 9):
    cache_cpu[0, page, 0, 0, 0] = 12
    cache_cpu[0, page, 0, 0, 1] = 100
    cache_cpu[0, page, 1, 0, 1] = -212
cache = cache_cpu.npu()[0]
rope_cache = torch.zeros((1, 209, 128, 1, 64), device="npu", dtype=torch.bfloat16)[0]
indices = torch.full((16, 1, 2048), -1, device="npu", dtype=torch.int32)
indices[:, 0, 0] = 0
indices[1::2, 0, 1] = 1
partial, lse = forward_dcp_sparse_attention(
    q_nope=query, q_rope=query_rope, k_nope=cache, k_rope=rope_cache, topk_indices=indices,
    actual_seq_lengths_query=torch.arange(2, 17, 2, device="npu", dtype=torch.int32),
    forward_metadata=SimpleNamespace(
        dcp_block_tables=torch.arange(1, 9, device="npu", dtype=torch.int32).view(8, 1),
        dcp_seq_lens=torch.full((8,), 2, device="npu", dtype=torch.int32),
    ),
    forward_batch=SimpleNamespace(forward_mode=ForwardMode.EXTEND),
    speculative_num_draft_tokens=None, scaling=1 / 16,
)
torch.npu.synchronize()
logits = torch.tensor([0.75, 0.0])
probabilities = torch.softmax(logits, dim=0)
exponentials = (logits - logits.max()).exp()
values = torch.tensor([100.0, -212.0])
print(json.dumps({
    "torch_npu": torch_npu.__version__, "shape": [16, 32, 512], "batch": 8,
    "logits": logits.tolist(), "values": values.tolist(),
    "fp32_pv_then_bf16": float((probabilities @ values).bfloat16()),
    "normalized_bf16_probability_pv": float((probabilities.bfloat16().float() @ values).bfloat16()),
    "unnormalized_bf16_exponential_pv": float(((exponentials.bfloat16().float() @ values) / exponentials.sum()).bfloat16()),
    "observed_column": partial.reshape(16, 32, 512)[1::2, 0, 1].float().cpu().tolist(),
    "observed_lse": lse[1::2, 0].float().cpu().tolist(), "expected_lse": float(torch.logsumexp(logits, dim=0)),
}, indent=2))
