"""Distinguish FP32 and BF16 softmax-probability inputs to the NPU PV product."""

import json
from types import SimpleNamespace

import torch
import torch_npu
from sglang.srt.hardware_backend.npu.attention.dsa_dcp import forward_dcp_sparse_attention
from sglang.srt.hardware_backend.npu.utils import init_npu_backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode

init_npu_backend()
torch.npu.set_device(0)
query = torch.zeros((1, 32, 512), device="npu", dtype=torch.bfloat16)
query[:, :, 0] = 1
query_rope = torch.zeros((1, 32, 64), device="npu", dtype=torch.bfloat16)
cache_cpu = torch.zeros((1, 2, 128, 1, 512), dtype=torch.bfloat16)
cache_cpu[0, 1, 0, 0, 0] = 12
cache_cpu[0, 1, 0, 0, 1] = 100
cache_cpu[0, 1, 1, 0, 1] = -100
cache = cache_cpu.npu()[0]
rope_cache = torch.zeros((1, 2, 128, 1, 64), device="npu", dtype=torch.bfloat16)[0]
indices = torch.full((1, 1, 2048), -1, device="npu", dtype=torch.int32)
indices[0, 0, :2] = torch.tensor([0, 1], device="npu", dtype=torch.int32)
partial, lse = forward_dcp_sparse_attention(
    q_nope=query, q_rope=query_rope, k_nope=cache, k_rope=rope_cache, topk_indices=indices,
    actual_seq_lengths_query=torch.tensor([1], device="npu", dtype=torch.int32),
    forward_metadata=SimpleNamespace(
        dcp_block_tables=torch.tensor([[1]], device="npu", dtype=torch.int32),
        dcp_seq_lens=torch.tensor([2], device="npu", dtype=torch.int32),
    ),
    forward_batch=SimpleNamespace(forward_mode=ForwardMode.DECODE),
    speculative_num_draft_tokens=None, scaling=1 / 16,
)
torch.npu.synchronize()
probabilities = torch.softmax(torch.tensor([0.75, 0.0]), dim=0)
values = torch.tensor([100.0, -100.0])
print(json.dumps({
    "torch_npu": torch_npu.__version__, "logits": [0.75, 0.0], "values": [100.0, -100.0],
    "fp32_probabilities": probabilities.tolist(), "bf16_probabilities": probabilities.bfloat16().float().tolist(),
    "fp32_probability_pv_then_bf16": float((probabilities @ values).bfloat16()),
    "bf16_probability_pv_then_bf16": float((probabilities.bfloat16().float() @ values).bfloat16()),
    "observed_column": partial.reshape(1, 32, 512)[0, :, 1].float().cpu().tolist(),
    "observed_lse": lse.float().cpu().tolist(), "expected_lse": float(torch.logsumexp(torch.tensor([0.75, 0.0]), dim=0)),
}, indent=2))
