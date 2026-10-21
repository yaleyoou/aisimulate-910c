# 910C + GLM-5.2 W8A8 算子级预测适配方案

整理日期：2026-10-20。111

本文根据本地 AISimulate 源码及 `/mnt/raid/user/data/models/glm-q` 模型文件整理。它是实施方案与源码差距评估，不代表已完成硬件适配或实测验证。

第一阶段的实物核验与新版本部署进展见 [第一阶段基线报告](910c-glm52-w8a8-stage1-baseline.md)。

单机硬件微基准的脚本、两轮实测和质量限制见 [单机采集报告](910c-single-node-collection.md)。

## 1. 评估结论

可以按以下四阶段推进，依赖顺序合理：

1. 固定模型、软件、硬件及量化基线。
2. 打通算子采集、性能数据库及整轮 forward 预测。
3. 对齐调度和 KV 管理，验证完整请求回放。
4. 若目标需要，再扩展 PD 分离。

需要补强四点：实际量化解析与类型支持、目标运行时的 NPU 采集、提前验证内存可行性、明确支持域和验收标准。添加一个硬件 YAML 并导入耗时表，不足以完成适配。

首版建议限定：固定软件版本和并行配置，PD 合并，关闭 MTP，先分别验证纯 prefill 和纯 decode，再验证 8192 输入 token、1024 输出 token 的完整请求。

优先考虑单节点，但必须以能实际加载模型和运行目标请求为前提。最小版本不等于单卡；如果单节点容量不足，应先调整首版部署范围。

## 2. 已确认的模型事实

已读取 `config.json`、`quant_model_description.json`、权重索引，并抽查部分 safetensors 文件头；没有加载完整模型权重或执行推理。

| 项目 | 本地配置 |
| --- | --- |
| 架构 | `GlmMoeDsaForCausalLM` |
| 主干层数 | 78 |
| FFN 结构 | 3 个 dense 层、75 个 sparse/MoE 层 |
| Hidden size | 6144 |
| Routed experts | 256 |
| 每 token 选择的 routed experts | 8 |
| Shared experts | 1 |
| MoE intermediate size | 2048 |
| DSA indexer | 21 层 `full`、57 层 `shared` |
| Index top-k | 2048 |
| 附加预测层 | 配置包含 1 个 next-token prediction 层 |

本地配置与仓库 `zai-org--GLM-5.2_config.json` 比较，仅发现 `moe_router_dtype` 字段存在差异。配置相近不代表量化方式、运行时执行和内存布局相同。

### 2.1 实际量化映射

| 抽查的权重 | 量化描述 | 文件存储 dtype |
| --- | --- | --- |
| `self_attn.q_a_proj.weight` | `W8A8` | INT8 |
| `self_attn.kv_b_proj.weight` | `FLOAT` | BF16 |
| `self_attn.indexer.wk.weight` | `FLOAT` | BF16 |
| dense MLP `gate_proj.weight` | `W8A8_DYNAMIC` | INT8 |
| routed expert `gate_proj.weight` | `W8A8_DYNAMIC` | INT8 |
| shared expert `gate_proj.weight` | `W8A8_DYNAMIC` | INT8 |
| MoE router `gate.weight` | `FLOAT` | BF16 |
| 附加预测层的 `q_a_proj.weight` | `FLOAT` | BF16 |
| `lm_head.weight` | `FLOAT` | FP32 |

以上是文件存储精度。运行时是否转换、实际计算精度、量化 scale/offset 的处理、KV cache 精度及融合方式，需要另行核实。量化描述还包含旋转相关元数据，应确认相应转换或开销是否进入执行与采集边界。

本地 README 标题为 W8A8，但示例量化命令含 `w4a8c8`，存在歧义。以实际权重、量化描述和运行时执行为准，不能仅凭模型名称推导所有精度。

## 3. 已有能力与主要缺口

| 项目 | 当前情况 | 适配动作 |
| --- | --- | --- |
| 模型识别 | 已注册 GLM DSA 架构，映射到 `DEEPSEEKV32` | 复用并验证目标执行路径 |
| MLA/DSA 几何 | 已有 GLM 对应维度 | 对照 Ascend 融合模块和 cache 布局 |
| Indexer 共享 | 已有 21/78 full 比例逻辑 | 核对目标运行时，分别采集 full/shared 路径 |
| Dense 前缀 | 特定部署条件下已有支持 | 检查选定并行配置是否进入准确建模路径 |
| 量化解析 | 未找到 `quant_model_description.json` 的解析路径 | 增加模块级量化映射 |
| GEMM INT8 | `Sq` 模式使用 INT8 计算 dtype | 验证静态/动态 W8A8 语义是否足够，不足则扩展 |
| MoE INT8 | MoE 枚举没有对应 INT8 W8A8 模式 | 补充类型、查询键、读取和计算语义 |
| 910C 系统与数据 | 所检查系统路径未发现对应条目 | 增加有来源的设备定义和实测数据 |
| 算子 Collector | 检查的 SGLang GEMM collector 使用 CUDA | 增加 NPU 执行、计时、同步与通信采集 |
| 请求回放 | 已有 SGLang 调度/KV 框架 | 核对 SGLang-Ascend 版本差异 |

已有 dense 前缀建模受后端、PP、CP、Attention-DP、MoE 通信和 nextn 等条件限制。模型名称可以识别，不代表所有拓扑都准确支持。

新增能力应贯穿 Python/Rust 类型、配置、序列化、数据选择、诊断和 Replay，继续使用统一入口：

```python
RustForwardPassPerfModel.best_available(config)
```

对应 Rust 的 `ForwardPassPerfModel::best_available(config)`。不要另建一套只能通过独立脚本调用的预测接口。不能用 FP8 标签或数据冒充 INT8 W8A8。

## 4. 第一阶段：基线、设备口径与模型身份

### 4.1 软件和模型基线

产出一份可复现版本清单及启动命令，至少包括：

- 权重路径、配置与量化描述的内容标识，以及可靠的权重版本或不可变快照证据。
- SGLang-Ascend 提交或镜像 digest、实际启动参数。
- PyTorch、torch-npu、CANN、驱动、HCCL、相关算子包或扩展版本。
- TP/EP/DP/CP/PP、MTP、调度预算、执行模式和容量参数。

先区分“计划使用 SGLang-Ascend”和“已经在该构建上运行成功”。本地 README 的 vLLM-Ascend 描述不能替代 SGLang-Ascend 验证。

### 4.2 910C 硬件定义

硬件 YAML 参数需来自目标机器、可靠规格或受控实测，明确：

- 实际设备型号和运行时可见设备口径。
- 模拟器一个 device 对应的资源范围。
- 单 device 可用内存、相关精度计算能力、内存带宽。
- 节点内互联分组、带宽与延迟；跨节点另行扩展。
- 权重、工作区、通信和图执行等预留。

不能混用板卡总容量和运行时单 device 容量，也不能复制 H200 的经验系数作为 910C 参数。

### 4.3 量化映射与容量预检查

保留模块级静态/动态量化、scale/offset、未量化模块及执行 dtype。检查 MLA cache、DSA indexer cache、量化辅助张量和运行时工作区。

第一阶段就应验证固定部署能初始化并运行目标 8k/1k 请求。完整 KV 调度仿真可以后做，但内存可行性不能推迟到第三阶段。

**验收：** 版本清单和启动命令可复现；设备口径明确；目标模型可运行；量化映射明确；未确认字段被显式记录。

## 5. 第二阶段：算子采集、数据库与 forward 预测

```text
实际算子和融合边界
  → shape/dtype/量化/并行采集 case
  → 910C 预热、同步、重复测量
  → 带执行身份的性能数据库
  → 模型和 batch 推导形状
  → 查询或插值延迟
  → 按层数、通信及有证据的重叠关系组合
  → 与独立整轮 forward 实测对照
```

### 5.1 算子映射与采集

至少核对：静态和动态 W8A8 GEMM、保留精度的矩阵、MLA/DSA、full/shared indexer、MoE dispatch/专家计算/combine、HCCL collective，以及必要的转换和数据搬移。

相同实现、shape、精度和执行条件可以跨层复用数据。融合模块不能再叠加内部算子耗时；重叠折扣需要执行证据。

MoE 采集应考虑专家负载分布。相同总 token 数不一定有相同的专家计算和通信成本。

### 5.2 数据和类型接口

采集需明确预热、异步同步、计时边界、重复次数、波动和异常处理；目标采用图执行时，应对齐实际执行方式。

数据需携带硬件口径、软件构建、算子实现、量化方式、并行配置、形状、计时单位和来源。

不一定必须新增 `sglang_ascend` 顶层 backend；可以评估复用 `sglang` 调度类别，但配置、选表和来源必须可靠区分 NPU 与 CUDA 执行实现。

### 5.3 8k/1k 首版采样域

| 对象 | 覆盖内容 |
| --- | --- |
| Prefill 新 token | 实际出现的 chunk，如 1024、2048、4096、8192 |
| Prefill prefix | 配套已有 KV 长度，不能只采 prefix=0 |
| Decode batch | 从 1 开始，覆盖目标负载实际出现的 batch |
| Decode 上下文 | 覆盖 8k 到 9k 附近轨迹，统一是否含当前 token 的口径 |
| DSA | 完整 indexer、复用索引及形状触发的不同实现 |
| MoE | 实际专家布局、token 分布和并行切分 |
| 通信 | 实际 collective、rank 组和消息大小 |

先推导有效形状，不盲目做全维度笛卡尔积。保留未用于校准的点和代表性负载作独立验证。

先校验纯 prefill 和纯 decode，输出 attention、indexer、MoE、通信等分解，避免只看总时间而忽略误差抵消。

**验收：** 目标域有覆盖；缺数据明确失败；单算子未见点和整轮 forward 达到约定精度；融合、重复次数、重叠边界可解释。

## 6. 第三阶段：完整请求模拟

对齐请求到达、并发控制、准入、chunk、decode 优先级、实际 batch/prefix、图执行 padding，以及 KV 分配释放、容量边界、适用路径中的抢占和重算。

已有 SGLang 回放不自动等于目标 SGLang-Ascend 的执行语义。检查调度参数是否真正传入原生调度器，同时核对混合 prefill/decode 轮次的计时组合。

普通自回归且无 MTP 时，8192 输入、1024 输出的基线是 prefill 产生首 token，再执行 1023 次 decode。并发和分块会改变轮次、等待、batch 和上下文，不能用固定 decode 时间代替整个轨迹。

```text
TTFT = 首 token 时间 - 请求到达时间
TPOT = (末 token 时间 - 首 token 时间) / (输出 token 数 - 1)
E2E  = 末 token 时间 - 请求到达时间
输出吞吐 = 对应窗口的输出 token 数 / 窗口时长
```

**验收：** 匹配配置与负载下，请求完整完成；TTFT、TPOT、吞吐和 KV 行为与实测对齐。数据覆盖、调度正确性、实测准确率分别报告。

## 7. 第四阶段：按需扩展 PD 分离

额外验证 KV 的传输表示、indexer 状态传递或重建、P/D 布局转换、有效带宽、固定延迟、共享链路竞争、接收端预留、交接事件及传输计算重叠。

参数来自目标部署实测。PD 不属于首版目标时，不阻塞 PD 合并版本交付。

## 8. 验收标准与交付物

精度门槛应在验证前约定，并先确认测量波动。以下仅为可讨论目标，不是已取得结果或精度保证：

| 项目 | 建议 |
| --- | --- |
| Forward | 可讨论中位相对误差 ≤10%、相对误差 P95 ≤20% |
| 短算子 | 同时设置绝对时间误差门槛 |
| 服务指标 | TTFT、TPOT、吞吐独立设门槛并固定统计窗口 |
| 验证数据 | 含未见点，不只验证用于校准的数据 |
| 稳定性 | 重复测量并记录波动与异常原因 |
| 覆盖 | 报告缺失和回退，跳过的点不算通过 |

软件验收包括配置往返不丢失身份、Python/Rust 一致、已有硬件回归、缺数据明确诊断。还需检查模拟器自身成本：初始化后复用模型和数据索引，避免每个 decode token 重新解析或扫描数据。

交付物清单：

1. 模型与软件版本清单、可复现启动命令。
2. 910C 硬件定义及参数来源。
3. 量化映射表和解析/类型支持。
4. 逻辑操作到实际算子的映射及融合边界。
5. NPU Collector、case 和原始证据。
6. 性能数据库、查询接口和 forward 分解。
7. 单算子、整轮 forward、8k/1k 请求验证报告。
8. 可复现预测配置和明确的支持范围、限制。

正式运行日志和测量证据放在 `benchmarks/evidence/` 的独立运行目录。配置字段更新其所属现有文档页。派生外部代码时遵守仓库第三方来源、不可变版本、许可证及 NOTICE 要求。

## 9. 源码入口

路径相对于仓库根目录，行号以本次检查为准。

| 源码 | 关注点 |
| --- | --- |
| `python/aisimulate/src/aisimulate_core/sdk/common.py:865` | 架构映射 |
| `python/aisimulate/src/aisimulate_core/sdk/models/deepseek_v32.py:33` | Indexer 共享 |
| `python/aisimulate/src/aisimulate_core/sdk/models/deepseek_v32.py:217` | 模型创建、量化、dense 前缀条件 |
| `python/aisimulate/src/aisimulate_core/sdk/utils.py:1613` | 本地量化配置读取 |
| `crates/core/src/perfmodel/common/enums.rs:206` | GEMM 精度 |
| `crates/core/src/perfmodel/common/enums.rs:300` | MoE 精度 |
| `crates/core/src/perfmodel/perf_database/dsa.rs:228` | GLM DSA 几何 |
| `python/aisimulate/collector/sglang/collect_gemm.py:283` | CUDA GEMM Collector |
| `crates/core/src/perfmodel/fpm/model.rs:405` | 统一构造入口 |
| `crates/core/src/engine/scheduler/sglang/config.rs:10` | SGLang 调度配置 |
| `python/aisimulate/.claude/rules/perfmodel-api.md` | 性能模型修改约束 |
| `docs/perf-model/fpm-self-service/implementation.md` | 正式接入与 checkpoint 工作流 |

四阶段是项目实施分解。正式 FPM onboarding 仍须遵循仓库六阶段流程、checkpoint 持久化和每配置 profile 接受规则，本文不替代这些记录。

## 10. 当前进度与下一步

已完成源码与模型元数据检查、部分权重头核对、初步差距识别。尚未完成目标运行时启动、硬件实测、适配代码、采集、导入和精度验证；没有已完成的 onboarding checkpoint。

环境检查：当前 shell 的 Python 为 3.9，项目要求 `>=3.11,<3.14`；所检查位置没有项目 `.venv`，`aisimulate` 与 `uv` 不在当前 PATH。尝试从 checkout 执行 `onboard --help` 因缺少 `psutil` 失败。这不代表机器上不存在其他环境。

下一步优先形成固定部署版本/启动清单、真实量化映射、逐项源码差距表。先确认目标运行时能运行本地权重，再确定采集与实现范围。

## 附录：md 目录为什么没有显示绿色

检查时，仓库配置了 GitHub origin `yaleyoou/aisimulate-910c`。Git 显示 `?? md/test.md`，说明已经识别到未跟踪文件；未发现目录被忽略规则排除。

是否显示绿色取决于编辑器、主题、Git 状态装饰、仓库识别和刷新，不能凭颜色判断目录是否属于仓库。本次未确认编辑器类型，无法断定具体 UI 原因。

Git 跟踪文件，不单独跟踪空目录。可使用：

```bash
git status --short --untracked-files=all -- md
git check-ignore -v md/test.md
```

`??` 表示未跟踪；`check-ignore` 无输出通常表示没有匹配忽略规则。`git add` 只暂存文件，commit 才进入本地提交，push 才上传远程；创建文件不会自动完成这些操作。

另有一个与颜色不同的权限问题：此前沙箱内显示 `md/` 属于 `nobody:nobody`，当前用户没有目录写权限；沙箱外检查确认目录实际属于 root，且 sudo 正常。经用户授权，已将目录所有者改为 `youxingye`，并将本文移入 `md/`。现有 `md/test.md` 保留不变。
