# 910C + GLM-5.2 W8A8：第一阶段基线核验

核验日期：2026-10-20。AISimulate 源码基线：`c9e7363bf5983b980fe6622f7b064ee7adff5f5d`，分支 `main`。

本报告对应 [适配计划](910c-glm52-w8a8-adaptation-plan.md) 的第一阶段，证据保存在 [stage1](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/)。

## 1. 结论与范围

已核对模型配置、量化描述、全部权重分片 header、主机设备和你的部署脚本。第一阶段固定具体的预测对象：模型、量化方式、设备口径、运行时和启动配置。算子库建设、预测误差验收及 PD 传输适配属于后续阶段。

你进一步要求使用最新 SGLang NPU 并沿用 United，因此已部署并验证 **SGLang v0.5.22 / CANN 9.1.0 / A3 镜像**。服务 `glm52-united-v0522` 正在运行，健康检查、完整短问答以及 8192 输入 / 1024 输出请求均已通过。启动结果见第 7 节。

不能据此宣称 AISimulate 已完成适配：910C 性能参数、ModelSlim 模块级精度、MoE INT8、Ascend 算子采集和查询仍有缺口。

## 2. 软件与模型身份

### 2.1 模型文件

实际路径：`/mnt/raid/user/data/models/glm-q`。

| 项目 | 核验结果 |
| --- | --- |
| 架构 | `GlmMoeDsaForCausalLM` / `glm_moe_dsa` |
| config SHA256 | `817f5fb39ca5d4c4b5648de89ca00deaea7537d8c2f130172a459252a05c1073` |
| 量化描述 SHA256 | `3386f968cd7049fe95f896c1a1aeacaa5c1c0659ac2ed9a42cd783cc48ef29ba` |
| 权重索引 SHA256 | `dfa97fa50b5e675ff6cea6ddeae3110795b6b7e971e6dc9cf565a4005fcb079c` |
| 分片数 | 182 |
| 索引 / header tensor 数量 | 均为 176,588 |
| tensor payload 总大小 | 773,854,402,152 bytes，约 720.708 GiB |
| header / 索引结构检查 | 未发现缺失、重复 tensor 或分片映射冲突 |
| 完整内容校验 | 未逐字节哈希约 720 GiB payload；尚无不可变 checkpoint revision |

索引 `metadata.total_size` 比 header 汇总少 75,497,472 bytes，恰好等于 `rot.weight` 的 BF16 `[6144,6144]` 大小。此差异单独记录：既未证明权重损坏，也未证明 writer 有意排除了该 tensor。

详见 [model-inventory.json](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/model-inventory.json) 和 [weight-shards.json](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/weight-shards.json)。配置哈希不能替代整份权重版本，磁盘存储量也不是运行时每 rank 权重占用。

### 2.2 原部署与版本选择

- `/home/youxingye/United deployment/service/`：本次单节点、PD 合并基线。
- `/home/youxingye/glm52-pd-1p1d/`：后续 PD 参考；当前主要提供 decode 侧脚本，不能据此认定完整 P/D 链路已复现。

原 United 标签为 `cann9.0.0-a3-v0.5.16`，镜像内包版本却为 **`0.5.17.dev386+gc5bd3d7dc`**，源码提交为 `c5bd3d7dce7623b8d2ffe3e662d3fd5198e6f4ba`，另有两个打包文件变动。不能只用“0.5.16”标识该运行时。

2026-10-20 查询 GitHub release API 得到最新正式 release `v0.5.22`，发布时间 `2026-10-19T02:55:19Z`；Ascend 仓库存在 `v0.5.22-cann9.1.0-a3`。本次固定：

```text
quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443
```

查询来源、时间和返回值见 [upstream-version-check.json](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/upstream-version-check.json)。`main` 是另一个持续变动的开发目标；本次按最新正式版固定基线。镜像内部版本及兼容性仍单独核验。

主机驱动 `26.1.1`、Ascend HAL `7.35.23`、主机 HCCL 文件 `8.5.0`。新镜像内实际核验结果如下，见 [runtime-image-inventory.json](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/runtime-v0522/runtime-image-inventory.json)。

| 组件 | 镜像内实际版本 |
| --- | --- |
| SGLang | 0.5.22 |
| 源码提交 | `baa9befa381511cd4d56f83980951e5a45fffc6d`；另有 pyproject 打包文件变动，镜像 digest 才是完整运行时身份 |
| Python | 3.12.13 |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0.post6；通过 torch-npu 使用 NPU，不因 torch 带 cpu 后缀就判定没有 NPU 支持 |
| CANN / HCCL | 9.1.0 B243 / 9.1.0 |
| DeepEP | 1.0.0+0234a271.cann.9.1.0.b243 |
| Transformers | 5.19.0 |

容器版本与主机 toolkit 分别记录。小算子初始化已通过，但不代表完整模型或通信已验收。

## 3. 910C 设备口径

`npu-smi` 观察到 **8 板 × 每板 2 芯片 = 16 个芯片设备**。每芯片显示 `65536 MB` HBM，按该工具常见二进制容量口径约为 64 GiB，整机约 1 TiB。工具名称为通用的 `Ascend910`；product 查询返回不支持，商业 SKU 保留用户给定的 910C 标识。

建议预测器的一个 device 对应一个运行时可见芯片，物理容量约 64 GiB。不能同时按板卡设置容量、按芯片设置 TP16，否则会重复计算。新版 `torch.npu.device_count()` 已实测为 **16**，设备名称为 `Ascend910_9382`。

运行时 `total_memory` 为偶数设备 65,787,658,240 bytes、奇数设备 65,796,046,848 bytes，低于名义物理容量。设备 0 在小型 BF16 matmul 后 free memory 为 65,354,162,176 bytes；此值属于**模型加载前的探测**，不能作为服务 KV capacity。见 [device-probe.json](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/runtime-v0522/device-probe.json)。模型加载后的容量另见第 7 节。

同板 `(0,1)、(2,3)…(14,15)` 连接为 `SIO`；跨板为 `HCCS_SW`。有效带宽、固定延迟、DeepEP/HCCL 随消息大小变化的耗时待测。

[hardware-facts.yaml](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/hardware-facts.yaml) 是有来源的事实草稿，**不是可直接注册的有效 system YAML**。HBM 带宽、BF16/INT8 吞吐和通信系数没有可靠值，保持未解析；运行时可用显存和 KV capacity 也不能由名义 HBM 直接替代。

## 4. 模型结构与量化

### 4.1 结构

| 部件 | 实际配置 |
| --- | --- |
| 主干 | 78 层：前 3 层 dense，后 75 层 MoE |
| Hidden / dense intermediate | 6144 / 12288 |
| MoE | 256 routed experts、1 shared expert、每 token top-8 |
| Expert intermediate | 2048 |
| Attention heads | 64 |
| Q LoRA / KV LoRA | 2048 / 512 |
| QK no-RoPE / RoPE / V | 192 / 64 / 256 |
| DSA indexer | 32 heads × 128，top-k=2048 |
| Indexer | 21 层 full、57 层 shared |
| 共享规则 | `index_topk_freq=4`、`index_skip_topk_offset=3`；不能直接用 1/4 |
| 词表 | 154880 |
| 最大位置 | 1048576，区别于本次部署 context-length |
| MTP | checkpoint 有 1 层；原 United 未启用推测解码 |

### 4.2 W8A8 不等于所有算子都用 INT8

下表是 checkpoint 的声明和存储类型；加载后布局、计算 dtype、激活量化融合以及缓存类型需要运行时核对。

| 部件 | 权重 shape 示例（未按并行切分） | 声明 / 存储 |
| --- | --- | --- |
| `q_a_proj` | `[2048,6144]` | W8A8 / INT8 |
| `q_b_proj` | `[16384,2048]` | W8A8 / INT8 |
| `kv_a_proj_with_mqa` | `[576,6144]` | W8A8 / INT8 |
| `kv_b_proj` | `[28672,512]` | FLOAT / BF16 |
| `o_proj` | `[6144,16384]` | W8A8 / INT8 |
| Full indexer `wq_b` | `[4096,2048]` | W8A8 / INT8 |
| Indexer `wk` / `weights_proj` | `[128,6144]` / `[32,6144]` | FLOAT / BF16 |
| Dense gate/up、down | `[12288,6144]`、`[6144,12288]` | W8A8_DYNAMIC / INT8 |
| Routed/shared expert gate/up、down | `[2048,6144]`、`[6144,2048]` | W8A8_DYNAMIC / INT8 |
| Router | `[256,6144]` | FLOAT / BF16 |
| Norm | 按模块 | FLOAT / FP32 存储 |
| Embedding / LM head | `[154880,6144]` | BF16 / FP32 存储 |
| Rotation | `[6144,6144]` | BF16；另有 rotation 元数据 |

完整分组表见 [quantization-map.csv](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/quantization-map.csv)。静态与动态 W8A8 需要区分；量化、反量化、scale 和融合边界都影响采集。不能将 KV、Attention、Norm 或通信因为模型名而直接配置为 INT8。

## 5. United 基线及 8k/1k workload

| 参数 | 原 United 基线 |
| --- | --- |
| CLI 并行 | TP16、DP2、EP16；启用 DP attention / DP LM head |
| 对应预测器 | Attention TP8×DP2；MoE TP1×EP16；PP1、CP1 |
| 设备总数 | 16，而非 32 |
| Attention / quant / dtype | ascend / modelslim / bfloat16 |
| MoE | DeepEP auto；禁用 shared-experts fusion |
| mem-fraction-static | 0.90 |
| context-length | 180000 |
| max-running-requests | 64，rank-local 实际限额仍需核对 |
| chunked-prefill-size / max-prefill-tokens | 16384 / 32768 |
| Decode graph | full，max batch 32，capture 8/16/24/32 |
| MTP | 关闭 |

启用 DP attention 后，`attention_tp = tp_size / dp_size / attn_cp_size = 8`；`moe_tp = tp_size / ep_size / moe_dp_size = 1`。两个 Attention DP 组共同参与 EP16，不是两套互不通信的完整模型副本。

**新版已经实际输出的生效配置有两处需要特别处理：**

- 旧 `--dp-size 2 --enable-dp-attention` 被规范化为 `dp_size=1, attn_dp_size=2`；新的 Attention DP 参数是 `--attn-dp-size 2`。旧参数目前兼容，但会发出弃用提示。
- DP attention 将 CLI 的 `chunked-prefill-size=16384` 除以 2，每个 Attention DP 组生效的 chunk 是 **8192**。page size 为 128；prefill graph disabled；decode graph full，capture 8/16/24/32。
- CLI 的 `max-running-requests=64` 对应 scheduler 日志中的每组 **32**。KV capacity 每组 **52992**，不能乘以 Attention TP8 当作独立 token 容量。

这些值来自 [runtime-resolved-args.json](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/runtime-resolved-args.json)，不是推测。迁移到预测器时应明确哪个字段表达全局预算、哪个表达 rank-local 预算，避免重复除 DP。

8k/1k 明确定义为 **8192 输入 token + 1024 输出 token**。单请求输入恰好等于当前每组 prefill chunk 8192，但每轮实际 shape 仍由并发、分组、预算和缓存决定。首 token 通常由 prefill 后采样产生，后续 decode 推进；不能把 1024 输出机械理解为 1024 次完全相同的 decode forward。Graph padding、DP 分配、top-k 共享、EP token 分布都需要进入 shape 推导。

更改 context、chunk、DeepEP 模式或 graph capture 时要记录新配置身份；旧测量不能直接当作新配置真值。

## 6. AISimulate 源码支持与缺口

| 环节 | 现状 / 后续工作 |
| --- | --- |
| GLM 架构 | `sdk/common.py` 已识别并映射 DEEPSEEKV32 |
| DSA / indexer | `sdk/models/deepseek_v32.py` 等已有 GLM 维度及 full/shared 逻辑，核对 Ascend 融合执行 |
| Dense 前缀 | 准确建模分支受 DP、MoE 通信等条件限制，不能默认覆盖 United DP2 + DeepEP |
| ModelSlim | 需要模块级 quant description 解析，区别静态/动态与 float 部件 |
| MoE INT8 | 现有 MoE 类型需扩展 W8A8 语义，贯穿配置、查询和计算 |
| Collector | 已检查 SGLang GEMM 使用 CUDA，需要 NPU 执行、计时和同步 |
| 910C | 缺有效 system spec 与配套算子库 |
| 自动 FPM onboarding | 本 checkout 的 `onboard init --help` 仅支持 `--framework vllm`，不能假填 vllm 代表 SGLang-Ascend |

已使用本 checkout 的 CLI 创建 [onboarding-checkpoint.json](../benchmarks/evidence/ascend910c-glm52-w8a8/onboarding-checkpoint.json)，记录真实调查和未完成项。主机 Python 3.9 不符合项目要求，CLI 在独立 Python 3.11 容器中执行。尚未生成、接受有效 FPM profile。

六阶段 onboarding 当前停留第 1 阶段调查；后续分别是运行时/范围选择、profile 审核、采集计划、采集质量与容量、replay/serving 验证。用户计划的“第一部分”与 CLI 阶段编号不是同一划分。

## 7. 新版本启动验证

**状态：启动与请求验证通过。** 2026-10-20 17:30 左右（北京时间）服务就绪；容器日志使用 UTC，即其中的 09:30。当前容器 `glm52-united-v0522` 正在运行，监听 `0.0.0.0:8000`，模型服务名为 `glm-5`。

| 验证项目 | 结果 |
| --- | --- |
| 原 United 参数 | 33 个原选项在新版 help 中均存在；弃用参数仍兼容 |
| 设备探测 | 16 NPU，设备 0 BF16 matmul 通过 |
| 权重加载 | 完成；运行时报告约 886.38 秒 |
| KV | BF16，每组 52992 token，page size 128 |
| Decode graph | 16 个 rank 全部完成，capture 8/16/24/32，约 149 秒 |
| Graph 内存 | 每 rank 日志约 4.27 GB；捕获后剩余约 1.79～2.10 GB，日志数值已四舍五入 |
| Scheduler | 每组 max-running-requests=32，chunk=8192 |
| 实际输入限制 | API 报告 max_req_input_len=52986，不能因为配置 180000 就认为可接收 180k prompt |
| 健康检查 | HTTP 200 |
| 短问答 | 两次 64-token 有限输出检查通过；额外一次完整问答正常 stop，236 输出 token，最终答案为 2 |
| 8192/1024 | prompt_tokens=8192、completion_tokens=1024、cached_tokens=0、无 retraction，HTTP 200 |
| 本次 8k/1k 时间 | 客户端 55.065 秒；服务报告首 token 4.762 秒、decode throughput 20.339 token/s |

容量和图证据见 [runtime-memory-observations.json](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/runtime-memory-observations.json)，请求结果见 [serving-smoke.json](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/serving-smoke/serving-smoke.json) 与 [完整短问答](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/serving-smoke/chat-complete.json)。原脚本未指定 reasoning parser，完整短问答中的思考段也包含在 content 中；此处只核验推理链路和最终输出，没有进行模型质量评估。

本次 8k/1k 用 tokenizer 产生合法 token 后重复、截取为精确 8192 个 token，设置 temperature=0、ignore_eos=true、max_new_tokens=1024。它是 **单请求功能验证**，不是正式性能统计，也不是随机 token 历史 benchmark 的等价重跑。不能由这一次结果判断新版相对旧版快慢。

新增启动入口：`/home/youxingye/United deployment/service/start-united-v0522.sh`。它固定新镜像并复用原 United 脚本，默认跳过主机系统调优，使用独立容器名；若该容器已存在则拒绝覆盖。

原参数首次启动后，权重加载持续约 25 分钟；进程栈停在 safetensors 读取/CPU 权重拼接，磁盘持续繁忙，尚未进入 forward。新版 loader 默认每 rank 使用 8 个加载线程。为验证是否为并发读盘问题，保留首次尝试容器 `glm52-united-v0522-attempt1` 及 [日志](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/attempt1/)，重试增加：

```text
--model-loader-extra-config '{"enable_multithread_load":false}'
```

仅关闭多线程后仍出现大量 mmap 页面等待，因此第二次尝试也已保留为 `glm52-united-v0522-attempt2`。第三次启动再增加 `--weight-loader-disable-mmap`，顺序读取分片，最终完成加载和服务验证。

为此对 `united-config.sh`、`run-united-container.sh`、`run-united.sh` 增加了可选 `MODEL_LOADER_EXTRA_CONFIG` 和 `WEIGHT_LOADER_DISABLE_MMAP` 的传递。原入口分别默认空值和 0；新版本入口默认关闭多线程并禁用 mmap。TP、DP、EP、图、调度、context 和精度均保持原参数。这只调整启动读权重策略；先前尝试也可能已经预热文件缓存，因此不能将加载速度差异全部归因于该参数。

```bash
bash '/home/youxingye/United deployment/service/start-united-v0522.sh' --check
bash '/home/youxingye/United deployment/service/start-united-v0522.sh' --start
docker logs -f glm52-united-v0522
```

启动入口需要 root 或已有的 sudo 授权。当前容器已存在时直接检查日志，无需再次执行 `--start`。部署快照见 [deployment-baseline.json](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/deployment-baseline.json)，原始日志见 [united-v0522-container.log](../benchmarks/evidence/ascend910c-glm52-w8a8/stage1/united-v0522-container.log)。已有 `sglang0520_npu_cryang` 容器未停止或修改。

这套容量对离线预测的影响很具体：假设各请求无共享前缀、每个请求都驻留完整 8192+1024=9216 token，则一个 DP 组的 52992 容量只能容纳 5 个这样的完整上下文，6 个需要 55296 token。它不是最大并发结论，因为真实请求处在不同阶段、可能共享缓存，还受预留和调度约束；它说明只设置 max-running=32 而忽略 KV 容量，会错估排队、回退和 TTFT。

## 8. 历史结果的用途

`United deployment/results/workload-concurrency-sweep_8k_1k/` 存有 **2026-09-18** 的原配置结果，未完整固定当时镜像 digest 和权重版本。

| 并发 | 完成请求 | TTFT ms | TPOT ms | 输出 token/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 3949.756 | 55.390 | 16.885 |
| 2 | 4 | 19012.898 | 56.044 | 26.814 |
| 4 | 8 | 30578.291 | 57.513 | 45.785 |
| 8 | 16 | 77955.828 | 64.250 | 56.994 |
| 16 | 32 | 156760.598 | 82.584 | 58.233 |
| 32 | 64 | 317410.646 | 102.575 | 63.892 |

这些是含排队、调度影响的端到端结果，不能写入算子数据库，也不能当作 v0.5.22 的误差验收。旧 PD 的容量记录不能替代当前 United KV capacity。

## 9. 后续验收

1. 已完成新运行时启动、请求验证和有效配置记录；仍需补齐权重不可变身份。
2. 补齐硬件系数、KV 容量、ModelSlim/INT8 MoE/NPU collector。
3. 采集明确的 prefill/decode batch，核对单算子和单轮 forward。
4. 再以同配置的 8192/1024 请求负载验证 TTFT、TPOT 和吞吐。

硬件事实、容量观察、采集质量、查询覆盖与端到端误差分别验收。服务健康检查通过不等于算子预测已准确。

第一阶段交付状态：**模型/量化调查、设备口径、版本清单、启动入口和实机服务验证已完成；可用于预测的 910C system spec 与模型精度/算子支持代码尚未完成。** `hardware-facts.yaml` 的空缺保持显式未解析。Checkpoint 中该配置仍保留第 1 阶段待补齐状态，下一步是形成有效的硬件与模型适配输入，再进入算子采集；没有把本次 serving smoke 标记为 FPM 精度验收。
