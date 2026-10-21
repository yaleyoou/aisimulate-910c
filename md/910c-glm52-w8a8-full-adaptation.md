# 910C + GLM-5.2 W8A8 完整适配目标

2026-10-22 启动持续 goal。目标是让当前模型通过 AISimulate 的统一算子级预测接口，
支持 PD 合并/分离、并行策略和 P/D 实例配比搜索。继续使用当前仓库，分阶段本地提交。

## 推进顺序与验收

| 阶段 | 主要工作 | 验收标准 |
|---|---|---|
| A：生产接口与缺口盘点 | 明确 W8A8 static/dynamic、HCCL、实际拓扑及数据键 | 不混精度、框架、驱动、通信后端和 graph 模式；缺数据明确报错 |
| B：补齐算子数据 | 实际模块与融合路径；路由、通信和 shape 网格 | 数值检查、独立重复、完整 case 清单和失败证据 |
| C：统一 FPE 接入 | Rust 算子计算/插值，Python 模型组合，内存和数据来源 | CLI、Sweeper、Replay 使用同一模型；原 GPU 回归保持正确 |
| D：真实部署验证 | United 多 batch/上下文，双节点 PD KV 传输 | forward 分解与实测对照，TTFT/TPOT/吞吐误差报告 |
| E：搜索 | 可行 TP/attention-DP/EP/PP、P/D 实例数 | 同一 workload/SLO 下给出可复现排序，并标明实测覆盖与外推 |

完整 case 缺口和计划保存在
`benchmarks/evidence/ascend910c-glm52-w8a8/production/coverage-plan.json`。
已有 36 个独立探针和真实 United 轨迹作为依据；它们不能单独支撑任意部署搜索。

## 两节点怎样使用

本机 npu1-32 和 npu1-19 各有 16 个运行时 NPU device。优先在空闲节点补齐单机数据，
需要跨节点通信或 PD 时联合运行。保留用户已有容器，不擅自删除其他服务。

两台机器的驱动分别为 26.1.1 和 25.5.0，数据需保留各自来源，验证可迁移性后才能共用。
720.708 GiB 是 checkpoint tensor 存储量，含 MTP 等特殊部分，不等于部署后的 resident memory。
搜索前要逐配置核算权重、量化 scale、KV/indexer cache、通信和 graph workspace。
不能因为总计 32 个 device，就假设任何 TP 或任意 1P:2D 都能实际部署。

## 数据真实性要求

- HCCL 用独立文件和精确拓扑键，不能把它写进 NCCL 数据。
- 模型中的 W8A8 static、dynamic、BF16 部分分别建模，融合内部操作不重复计时。
- DeepEP 的实际 expert token 分布、空专家、graph padding 和有效 KV 长度必须进入对应数据身份。
- 不以成功启动或 profiler 样本代替 forward/请求预测误差验证。
- 搜索“最优”只对声明的资源预算、workload、SLO 和已支持的策略空间成立。

正式六阶段 FPM 自助流程仍保存于原 `onboarding-checkpoint.json`；当前请求明确授权生产
producer/consumer 的协同适配，但不会把此授权伪造成某份 direct-FPM profile 已被精确接受。
