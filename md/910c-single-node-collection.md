# 910C 单机硬件采集：实现、结果与使用方法

## 1. 本次交付范围

本次实现并执行的是 **910C 单节点硬件微基准采集**，服务于 GLM-5.2 W8A8 的适配准备。一个模拟器 device 对应一个运行时可见芯片；当前节点共 16 个 device，来自 8 块双芯片板卡。

采集内容包括：

- 16 个芯片分别测量设备内存容量、128 MiB/512 MiB 显存复制、4096×4096×4096 BF16 和 INT8 矩阵乘。
- device 0 额外测量 1 GiB 显存复制、8192³ 矩阵乘，以及 M=1/8/32/128/512/2048/8192、K=6144、N=2048 的矩阵乘。
- 五种单节点通信分组：同板 2 卡 `[0,1]`、跨板 2 卡 `[0,2]`、Attention 8 卡 `[0..7]`、另一 Attention 8 卡 `[8..15]`、完整 16 卡 `[0..15]`。
- 每个通信组测量 BF16 AllReduce、AllGather、ReduceScatter、AllToAll，输入大小分别为 4 KiB、64 KiB、1 MiB、16 MiB、64 MiB。

共 **81 个单芯片测试点 + 100 个通信测试点**。其中 100 个通信点追加了一轮完整复测，首轮数据全部保留。

这里的“单机采集完成”与“GLM 算子预测适配完成”是两个验收项。本轮没有运行 MLA/DSA/indexer、融合 MoE、DeepEP dispatch/combine 或完整 forward 采集，也没有把微基准结果冒充正式 FPM 数据。

## 2. 环境与计时口径

使用上一阶段验证过的同一镜像：

```text
quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443
```

SGLang 0.5.22、torch-npu 2.10.0.post6、CANN/HCCL 9.1.0；运行时芯片名 `Ascend910_9382`。详细来源见 [上一阶段基线](910c-glm52-w8a8-stage1-baseline.md)。

执行前暂停 `glm52-united-v0522`，通过 `npu-smi info` 确认设备没有残留进程。所有分组按顺序执行，不与模型推理或其他采集组重叠。另一容器 `sglang0520_npu_cryang` 没有被停止或修改。采集脚本结束时自动启动原 United 容器，并检查 `/health`。

| 项目 | 首轮 | 通信复测 |
| --- | --- | --- |
| 预热 | 每点 10 次 | 每点 50 次 |
| 一份样本包含的调用 | 20 次 | 200 次 |
| 样本份数 | 7 份 | 9 份 |
| CPU affinity | 不主动绑定 | 每 rank 8 个互不重叠的逻辑 CPU |
| 计时模式 | eager | eager |
| 单芯片结果 | 实测 | 引用首轮原文件，记录 SHA256；没有重新采集 |
| 通信结果 | 实测全部 100 点 | 重新实测全部 100 点 |

CPU 集合按可用 CPU 分区选择，并记录到 `*-affinity-rank*.json`。设备 PCI 信息中的 NUMA node 为 `-1`，因此没有声称该绑定实现了 NPU/CPU 的 NUMA 就近放置。

**单芯片计时：**用 NPU Event 测每批调用的设备时间，同时保留主机墙钟时间；除以调用数得到每次平均耗时。Eager 提交可能让小矩阵的设备时间包含等待主机提交的空隙，因此不能把所有小矩阵结果当成纯内核执行时间。

**通信计时：**每份样本前执行 barrier 和设备同步，计时范围包含连续调用以及最后的设备同步，不含前置 barrier。所有 rank 保存各自耗时，对同一份样本取最慢 rank，再计算中位数和变异系数 CV。该数值包含 Python、ProcessGroupHCCL 调度和设备工作，并非裸链路延迟。

**质量规则：**不删除慢样本，不自动跳过失败点。CV > 10% 的测试点单独列出，不能作为已经通过重复性验证的数据使用。该阈值是本轮硬件调查规则，不代表正式 FPM 采集门禁已经通过。

## 3. 单芯片实测结果

以下为 16 个芯片各自测试中位数的跨芯片统计；采用十进制 GB/s、TFLOP/s 和 TOPS。矩阵乘按 `2×M×K×N` 计算操作数。

| 指标与测试形状 | 跨芯片中位数 | 芯片间范围 |
| --- | ---: | ---: |
| 512 MiB 显存复制有效读写带宽 | 1269.03 GB/s | 1266.69–1270.23 GB/s |
| BF16，M=K=N=4096 | 321.19 TFLOP/s | 318.28–322.57 TFLOP/s |
| INT8，M=K=N=4096 | 553.07 TOPS | 543.25–562.27 TOPS |
| 运行时总内存，较小值 | 65,787,658,240 bytes/device | 偶数/奇数芯片略有差异 |

这三组主指标在全部 16 个芯片上的重复样本 CV 都不超过 10%。81 个单芯片测试点均完成，正确性校验通过。

芯片计算/复制按 device 顺序执行，测的是各芯片独立运行时的基线；没有测试 16 片同时满载的计算吞吐，也没有将这些数值乘以 16 宣称整机吞吐。

显存复制的有效流量按“一次读取 + 一次写入”计算，即 `2×buffer_bytes/time`。这不是硬件计数器测得的物理 HBM 流量。主指标固定采用 512 MiB 缓冲，避免把更小缓冲的更高结果直接当作持续带宽。

INT8 使用 `torch_npu.npu_quant_matmul`，输入 INT8、每输出通道单位 scale、输出 BF16；不包含激活量化。这是矩阵计算基线，不能替代 ModelSlim 静态 W8A8、动态 W8A8 或融合 MoE 的端到端耗时。BF16 使用 `torch.mm`；权重为 ND 格式 `[N,K]` 的转置视图。

正确性检查包括显存复制的全缓冲字节相等，以及每个 GEMM 至多九个分散输出位置与 CPU FP32 点积参考比较。后者是数值抽查，不是全矩阵精度认证。

## 4. 通信结果与稳定性

首轮 100 个通信点全部完成，结果校验通过；其中 16 点 CV 超过 10%。首轮全部 181 个点中有 165 点满足本轮重复性阈值。

首轮 64 MiB 输入的 AllReduce 示例：

| 分组 | 中位耗时 | 输入字节数/耗时 | CV |
| --- | ---: | ---: | ---: |
| 同板 2 卡 `[0,1]` | 611.50 μs | 109.75 GB/s | 0.55% |
| 跨板 2 卡 `[0,2]` | 689.90 μs | 97.27 GB/s | 4.32% |
| 8 卡 `[0..7]` | 687.09 μs | 97.67 GB/s | 10.96%，未通过 |
| 8 卡 `[8..15]` | 687.67 μs | 97.59 GB/s | 0.21% |
| 16 卡 `[0..15]` | 920.45 μs | 72.91 GB/s | 3.71% |

**表中的 GB/s 是每 rank 的输入大小除以耗时，不是 SIO/HCCS 物理带宽，也不是 NCCL 的 bus bandwidth。**例如 AllGather 的输出大小是输入的 rank 数倍，ReduceScatter 的输出是输入的 rank 数分之一；不同 collective 的通信流量不能套用同一个解释。

追加复测的 100 个通信点也全部完成、正确性校验通过，但其中 **30 点** CV 超过 10%；不能把这轮称为全面改善。它引用首轮的 81 个稳定单芯片点，因此合计 151/181 点满足该轮阈值。

复测中，五种分组的 64 MiB AllReduce 中位耗时分别为 600.44、677.94、667.24、669.03、893.90 μs，该组坐标的 CV 均不超过 10%。完整曲线中其他坐标仍有波动，不能将这个局部结果推广到所有消息大小。

扩大采样窗口并固定 CPU 集合，没有消除这条 eager 测量路径的波动。不能据此认定是 HCCL 内核、链路或 CPU 的单一问题；也不能从两轮里挑每个点较好的结果拼成“全部通过”的数据集。

两轮的 `summary.json` 分别列出全部未通过重复性阈值的点；`cases.csv` 提供耗时、CV 和明确口径的吞吐。后续正式通信建模需要进一步对齐实际服务中的通信实现、执行图与重叠方式，并对不稳定坐标完成复核。

原始通信 JSON 的 `kernel_source` 用的是 collective 家族简称。实际调用以冻结脚本为准：`all_reduce`、`all_gather_into_tensor`、`reduce_scatter_tensor`、`all_to_all_single`，backend 均为 HCCL。当前脚本已将这个元数据字段改成完整 API 名；两轮原始数据和冻结脚本保持原样，耗时没有改写。

## 5. 文件与复现方式

脚本目录：

```text
benchmarks/evidence/ascend910c-glm52-w8a8/single-node/
```

| 文件 | 用途 |
| --- | --- |
| `collect.py` | 芯片计算/内存及 HCCL 采集；正确性校验；原始样本保存 |
| `run.py` | 固定镜像、独占预检、停止/恢复指定服务、串行运行所有分组 |
| `affinity.py` | 通信复测的 rank CPU 绑定，记录实际 CPU 集合 |
| `summarize.py` | 校验样本与 rank 覆盖，生成 CSV 和质量报告 |
| `checkpoint.py` | 通过当前 checkout 的 `aisimulate onboard` 恢复/更新检查点 |
| `hardware-measured.json` | 带来源和单位的硬件实测摘要；不是生产系统配置 |
| `run-20261020-231954/` | 首轮 81+100 点，包括原始样本、命令、日志、源文件快照 |
| `refined-20261020-2330/` | 100 点通信复测及明确引用的首轮单芯片数据 |

以下命令在仓库根目录执行。每次必须选择新的输出目录，不覆盖已有实验。

仅查看采集计划，不停止服务：

```bash
python3 benchmarks/evidence/ascend910c-glm52-w8a8/single-node/run.py \
  --output benchmarks/evidence/ascend910c-glm52-w8a8/single-node/my-new-run
```

执行完整采集，会暂时中断 United 服务：

```bash
python3 benchmarks/evidence/ascend910c-glm52-w8a8/single-node/run.py \
  --output benchmarks/evidence/ascend910c-glm52-w8a8/single-node/my-new-run \
  --execute
```

检查并汇总：

```bash
python3 benchmarks/evidence/ascend910c-glm52-w8a8/single-node/summarize.py \
  benchmarks/evidence/ascend910c-glm52-w8a8/single-node/my-new-run
```

复现本轮追加通信测量：

```bash
python3 benchmarks/evidence/ascend910c-glm52-w8a8/single-node/run.py \
  --output benchmarks/evidence/ascend910c-glm52-w8a8/single-node/my-new-communication-run \
  --reuse-local-source benchmarks/evidence/ascend910c-glm52-w8a8/single-node/run-20261020-231954/single.json \
  --rank-affinity --warmup 50 --iterations 200 --repeats 9 --execute
```

脚本依赖 Linux、Docker、主机 Ascend 驱动、已经存在的固定镜像、16 个可见芯片和 `bond4` 接口。它是针对当前机器的可复现调查工具，尚未包装成 AISimulate 通用 Ascend Collector 后端。重用本地数据只适用于同一硬件和软件基线。

运行后可查看 `manifest.json` 的 `status`、`service_restored` 和时间字段。采集退出与服务恢复是独立状态，`collected` 本身不代表服务已 ready。服务加载需要时间，以 `/health` 实际返回为准。

本次在 **2026-10-20 23:41:50（北京时间）**确认 United 恢复健康，两轮 manifest 的 `service_restored` 均为 `true`。第一轮恢复期间，为追加通信复测重新暂停了尚在加载的服务；最终恢复状态以第二轮 manifest 的时间线为准。

恢复后再次执行 8192 输入 / 1024 输出请求：`cached_tokens=0`、`num_retractions=0`、首 token 2.591 秒、客户端总耗时 51.585 秒，输出达到 1024 token。证据见第二轮目录的 `restored-serving-check.json` 和 `restored-response-8192-1024.json`。这是一次恢复功能检查，不是重复性、预测误差或性能提升认证。

本轮采集器以脱离终端会话的子进程运行，`active-run.json`、`active-refinement.json` 保存进程与日志位置。聊天流断开不会自动停止这个进程；恢复会话后应先检查 manifest、进程和服务状态，避免重复启动另一个采集任务。主机重启或进程被强制终止仍需要人工检查，不能把 `finally` 当作跨重启恢复保证。

## 6. 与 AISimulate 适配的关系

本轮补齐了单机硬件的可复现测量链路和原始证据，初步解决了之前“HBM/BF16/INT8 完全没有实测来源”的问题。

仍然需要完成：

1. 将实测证据整理为适用范围明确的 910C 硬件定义。实测有效吞吐与理论峰值不可混用，也不能再叠乘其他硬件的经验折扣。
2. 解决 HCCL 通信数据接入与稳定性问题。当前 AISimulate 的 NCCL/OneCCL 数据路径不能靠改名充当 HCCL；单个 `intra_node_bw` 也不足以表达全部拓扑和集体通信曲线。
3. 按当前 SGLang-Ascend 实现，采集 ModelSlim 静态/动态 W8A8、融合 MoE、MLA/DSA/indexer 和 DeepEP 等真实算子边界。
4. 对齐 TP16、Attention DP2、EP16 的 shape 推导，先验证 prefill/decode 单轮 forward，再验证 8192 输入/1024 输出的完整请求。

跨节点带宽和跨节点 PD 不在本次范围，需要多节点实测。没有填入虚构的 `inter_node_bw`，没有生成看似可用但缺少依据的生产系统 YAML，也没有宣布模型适配或预测精度验证完成。

完整会话仍保存在 `benchmarks/evidence/ascend910c-glm52-w8a8/onboarding-checkpoint.json`；配置 `united-tp16-dp2-ep16` 的正式 FPM profile 尚未接受，阶段 1 的生产硬件/模型接入仍未完成。
