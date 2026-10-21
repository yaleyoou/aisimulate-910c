# 910C + GLM-5.2 W8A8：真实算子与采集范围

更新日期：2026-10-21。这里的“第 1、2 项”指硬件适配后的实际算子映射和采集，
不是 FPM 自助接入流程的第 1、2 阶段。硬件及前期基线已经提交为 `93011105`。

## 1. 已确认的运行对象

沿用正在运行的 United：SGLang 0.5.22，源码提交
`baa9befa381511cd4d56f83980951e5a45fffc6d`，ModelSlim W8A8，模型 `/models/glm-q`，
宿主机权重 `/mnt/raid/user/data/models/glm-q`。本阶段固定版本采集，不再用浮动 latest。
源码哈希和逐文件版本一致性记录见
`benchmarks/evidence/ascend910c-glm52-w8a8/operators/serving-source-inventory.json`。

- 16 个运行时 NPU device，TP16，attention DP2 / TP8，MoE EP16 / TP1。
- 78 层：前 3 层 dense MLP，后 75 层 MoE；每层 256 routed experts，top-8，另有 1 个 shared expert。
- DeepEP auto：prefill 与 decode 使用不同的通信执行路径；shared expert 独立执行且 TP1。
- BF16 activation / KV，静态 W8A8 attention linear，动态 W8A8 dense/shared/routed MLP。
- decode graph bucket 为 8/16/24/32，prefill graph 关闭，MTP 关闭。
- `GlmMoeDsaForCausalLM` 实际继承 `DeepseekV2ForCausalLM`，不能只看 `glm4_moe.py` 的普通 GLM 分支。

## 2. 8k 输入 / 1k 输出时，shape 怎么变化

单请求固定发往 attention DP0；每次清空缓存，实测返回 `prompt_tokens=8192`、
`completion_tokens=1024`、`cached_tokens=0`、`num_retractions=0`。
这些检查验证请求执行条件，不代表模型回答已经做独立质量评测。

Prefill 的 DP0 attention 输入 token 数是 8192；其 8 个 TP rank 各持有 8 个 attention heads。
Dense MLP 按全局 TP16 切中间维度；MoE 前按运行时布局分发 token，再按 top-8 路由到 EP16。
不能将每个 expert 的 token 数简单设为 `8192×8/256`：实际热点和每层路由不同。

Decode 的有效请求数为 1，但 attention 图实际使用 batch 8；跨 attention DP 收集后的
dense MLP token 维度可为 16；shared expert 的本地输入为 1。MoE low-latency buffer
的物理 shape 为 4096 行，但这不是实际参与 GEMM 的有效 token 数。
因此“所有 decode 算子 M=1”以及“GMM M=4096 就算满 4096 行计算”都不成立。

## 3. 映射表：逻辑计算、运行时算子和实测 shape

下表使用逻辑矩阵 `[M,K] × [K,N]`。实际 INT8 权重可能是 FRACTAL_NZ 等分块布局；
原始 profiler 表中的物理 shape 和 formats 均保留，不能把四维物理布局误当成模型维度。
次数按一个 active attention rank / 一轮 forward 描述；idle DP、不同 EP rank 和分阶段通信须分别统计。

| 逻辑部分 | 真实执行 / 精度 | 8192 prefill 与单请求 decode 的关键 shape | 次数或约束 |
|---|---|---|---|
| Q-A + KV-A | `npu_quantize` + `npu_quant_matmul`，INT8×INT8→BF16 | K=6144，N=2048+512+64=2624；M=8192 / 8 | 78；两个投影融合，只算一次 |
| Q-B | 同上 | K=2048，N=64/8×256=2048；M=8192 / 8 | 78 |
| QK norm / RoPE / cache | NPU / Triton 的 norm、RoPE、scatter 等 | Q latent=2048，KV latent=512，RoPE=64 | 按真实融合边界统计 |
| MLA Q 投影 | `npu_transpose_batchmatmul` 等 BF16 BMM | local heads=8，192→512 | 78；不是完整 `kv_b_proj` 另做一次 dense GEMM |
| DSA indexer Q | 静态 W8A8 linear | K=2048，N=32×128=4096；M=8192 / 8 | 21 full-indexer 层 |
| DSA indexer K / weights | 浮点 linear / norm / RoPE | K head=128，weights heads=32 | 不能标成 W8A8 |
| DSA top-k | `npu_lightning_indexer`，实测 BF16 Q/K/weights | `[8192,32,128]` / `[8,32,128]`；top-k=2048 | 21；其余 57 层复用索引 |
| Sparse attention | `npu_sparse_flash_attention`，BF16 | Q `[8192,8,512]` / `[8,8,512]`，另有 64 维 RoPE | 78；KV 是分页 buffer，不是完整有效长度 |
| MLA V 投影 | BF16 BMM | local heads=8，512→256 | 78 |
| O projection | 静态 W8A8 linear + attention TP 通信 | K=8×256=2048，N=6144；M=8192 / 8 | 78 |
| Dense gate+up | dynamic quant + INT8 GEMM | K=6144，N=2×12288/16=1536；M=8192 / 16 | 3；gate/up 合并 |
| Dense down | dynamic quant + INT8 GEMM | K=12288/16=768，N=6144；M=8192 / 16 | 3 |
| Router / top-k | 浮点 router、`MoeGatingTopK` | hidden=6144，experts=256，top-k=8 | 75 |
| Shared gate+up / down | 独立 dynamic W8A8 linear，TP1 | 6144→4096→2048→6144；本地 M=1024 / 1 | 75；不能与 routed GMM 重复计入 |
| Routed gate+up / down | `npu_grouped_matmul`，INT8 权重、逐 token scale，BF16 输出 | 本 rank 16 experts；权重 `[16,6144,4096]` 与 `[16,2048,6144]` | 每层 2 次，共 150；每层实际路由行数不同 |
| Routed activation | `_swiglu_quant_kernel` | gated hidden=2048，量化供 down GEMM 使用 | 75；避免再加一份独立 SwiGLU+quant |
| Prefill MoE 通信 | `CamMoeDispatchNormal` / `CamMoeCombineNormal`，含 layout/notify | 由实际路由决定 | 75；包含通信与整理开销 |
| Decode MoE 通信 | `MoeLowLatencyDispatchV2` / `MoeLowLatencyCombineV2` | 本地输入 `[1,6144]`，接收 buffer `[4096,6144]` | 75；buffer 容量不等于有效负载 |
| TP / DP collective | HCCL AllGather、ReduceScatter、AllReduce | 按 rank、通信组和 tensor 实测 | 不套用 NCCL 文件或 NVIDIA kernel |
| Embedding / lm_head / sampling | 浮点 gather/GEMM/sampling 等 | 与词表、DP lm_head 配置有关 | 原始采集保留，端到端组合不能漏掉 |

Full indexer 层为 0、1、2、6、10、…、74，共 21 层。来源是
`srt/configs/model_config.py:410` 的频率/offset 规则，且 profiler 确认每次 active forward
21 次 LightningIndexer、78 次 SparseFlashAttention。

## 4. 源码定位

以下路径均相对于上述固定 commit 的 `python/sglang/`。这里记录分析和调用关系，不复制上游实现。

| 文件与位置 | 对应事实 |
|---|---|
| `srt/models/glm4_moe.py:1289` | GLM DSA 继承 DeepSeek 实现 |
| `srt/models/deepseek_v2.py:2020` | attention TP heads、Q/KV-A 融合、Q-B / KV-B / O 分片 |
| `srt/models/deepseek_v2.py:800` | DeepEP shared expert 使用 TP1 |
| `srt/configs/model_config.py:410` | full/shared indexer 的实际层号规则 |
| `srt/hardware_backend/npu/quantization/linear_method_npu.py:1` | static / dynamic W8A8、量化参数和输出 dtype |
| `srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py:372` | DSA prepare、prefill/decode 分支、索引复用 |
| `srt/layers/attention/dsa/dsa_npu_indexer.py:229` | index KV dtype 分支；本部署实测 BF16 indexer |
| `srt/hardware_backend/npu/attention/ascend_backend.py:1471` | BF16 sparse attention 的真正接口和布局 |
| `srt/hardware_backend/npu/quantization/moe_methods.py:648` | INT8 MoE 权重重排、scale 和 dispatcher dtype |
| `srt/hardware_backend/npu/moe/matmul.py:23` | GroupedMatmul 的 group-list、scale、dtype 实参 |
| `srt/layers/moe/moe_runner/ascend.py:118` | MoE activation 与 GMM 融合决策 |
| `srt/layers/moe/token_dispatcher/deepep.py:1` | normal / low-latency 分发路径 |
| `srt/managers/scheduler_components/profiler_manager.py:145` | num_steps 的自动结束边界及 NPU profiler 路径 |

## 5. 采集和预测接入的边界

本阶段直接使用 United 的真实模型、真实权重和已选择的 Ascend kernels。原有 CUDA
collector 的 FP8 / FlashMLA / DeepGEMM 数据不能代替它们。

`collector/sglang/ascend_trace.py` 执行一次未开启 profiler 的对照，然后执行独立的多轮采集。
每轮请求完整生成 1024 token；只截取开头 3 次 forward，避免全程 profiler 造成大量数据和明显扰动。
`ascend_trace_report.py` 分 rank / graph ID / kernel / shape / dtype / layout 汇总，
只把三轮均出现且中位数 CV≤10% 的组标记为重复性合格。

这份采集能校验算子选择、次数、物理 shape、精度和局部耗时分布；它还不是可以直接喂给
AISimulate 插值器的完整独立算子数据库。原因包括：profiler 计时扰动、MoE group-list
内容未由 shape 表导出、decode 只覆盖开头上下文、通信父记录与内部 kernel 重叠。
不能把所有 kernel 或 CPU wrapper 时间相加当成 forward，也不能把请求执行成功当成
每个算子的独立数值正确性证明。

后续正式接入需要给这些 Ascend 特有维度建立生产查询契约，并单独完成代表性算子的
无 profiler 重复计时、数值检查、8k→9k 上下文覆盖及整轮 forward 误差验证。
FPM 自助流程目前没有接受本配置的精确 profile；此处不会将探索性观测伪装成已验收 FPM 数据。
