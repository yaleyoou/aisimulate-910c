# 910C + GLM-5.2 W8A8 生产适配进展

## 已完成的基础阶段

本阶段完成接口和模型元数据接入；完整算子预测、PD 和策略搜索仍在继续实现，不能据此宣称全部适配完成。

| 内容 | 已实现行为 |
|---|---|
| 原生 HCCL | Rust 独立算子与表读取，按 dtype、操作、rank 数、每节点 rank 数、布局和 eager/graph 区分；不套用 NCCL |
| 数据质量 | 逐 rank 前后全量正确性检查、完整样本与 CV 检查；不合格点保留原因；缺失配置和越界查询明确失败 |
| 运行设置 | 保存实际 United 环境、SGLang HCCL options 和工作进程 CPU 绑核；发布时核对一致性 |
| W8A8 | 区分静态 W8A8、动态 W8A8、weight-only INT8、SmoothQuant 和 FP8；MoE 使用动态 W8A8 身份 |
| 模型元数据 | 读取实际 ModelSlim sidecar，保留每层混合精度；确认78层、75层MoE、21层声明indexer |
| 准备开销 | 约19MB sidecar压缩为约33KB保留元数据，实测首次解析约0.145秒；复用现有配置缓存 |

最终基础版本的两套 GPU 预测回归共 **390 项通过**。另有原生 HCCL、wire schema、
source resolution、精度枚举及 Python 元数据/导出校验测试。296条通过质量门槛的
peer HCCL测量也已通过真实原生引擎读取，返回值与采集值逐点一致。

## 数据的适用范围

本次19节点完整运行设置下的HCCL网格覆盖2/4/8/16rank、4种collective、eager/graph和10个payload点，
共320点。其中296点通过CV≤10%的门槛，24点未通过，已保留，未混入合格表。
这份数据来自19节点的25.5.0驱动，不能自动当作本机26.1.1驱动的数据。

早期queue1/default-buffer控制实验、GC对照和任意8核绑核实验均保留为调查证据。
它们不是United的生产校准配置。实际United采用queue0、HCCL650MB、expandable segments及
记录下来的40个逻辑CPU/worker绑核。GC对照没有证实GC是波动原因。

本机同协议的36个合成算子控制实验数值与重复性检查全部通过，但与19节点并非所有shape都可互换。
尤其很小的独立动态量化case存在明显差异，故继续保留各自运行来源。

## 模型与下一步采集

196个模型文件、774079436873字节已复制到19节点独立私有目录。
两端逐文件SHA-256一致，整体内容标识为：

`b3645035a8b916c630ccab2f0d9be6b5971202e0ac47c9cf2d6dab63c1c6a9cc`

源目录是 `/mnt/raid/user/data/models/glm-q`；19节点目录是
`/mnt/raid/aisimulate-youxingye-20261022/glm-q`。现有其他用户容器没有被替换。

接下来采集真实模型每层、每个rank的expert token counts。Ascend的正常和低延迟MoE路径
使用 `group_list_type=1` 的实际计数，不能用固定buffer容量或均匀分配代替。
图回放计数复制已用真实NPU验证：输入从 `[2,2]` 改为 `[1,3]` 后，能读到新计数，
计算输出保持正确。该仪器只采状态，不用其耗时作为未插桩性能数据。

随后继续完成模型执行组合、完整算子/模块曲线、内存、整轮forward误差验证，再打通PD传输、
请求模拟和配置搜索。当前不宣称任意并行策略或任意P/D比例已有可靠预测。

原始记录、测试日志及数据表在
`benchmarks/evidence/ascend910c-glm52-w8a8/production/`。
持续goal与正式onboarding的状态仍保存在原 `onboarding-checkpoint.json`；精确FPM profile尚未接受。
