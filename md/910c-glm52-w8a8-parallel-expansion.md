# 并行策略扩展：对齐GPU版的实际能力

本轮目标是继续补齐并行策略、自动搜索和相关采集。此前EP16功能版是阶段交付，
不能代表完整并行适配完成。准确性细调仍按用户要求延后；新的执行方式、通信和KV布局属于功能工作。

## 当前核对结果

| 项目 | GPU版/目标运行时源码情况 | 当前适配需要完成的工作 |
|---|---|---|
| 默认搜索 | AISimulate默认枚举PP1、每worker1/2/4/8/16device；存在后端和模型约束 | 为910C模型提供自己的合法结构候选，保留原生内存检查和统一预测入口 |
| 副本与P:D比例 | 通用枚举可以按总预算生成副本组合 | 去掉对手写1/2副本预设的依赖，覆盖预算内1P:3D、3P:1D等组合 |
| EP32 | 模型组合已有，算子表大部分已采集，通信小batch范围有缺口 | 补通信范围并做整轮与请求预测验证 |
| Attention TP16、ADP1 | 相应线性/向量/注意力原语已有部分数据；执行边界与DP布局不同 | 观察真实运行时的padding、归一化、通信和logits路径后再接入 |
| 其他TP/EP、MoE TP | DeepEP会把EP调整为全局TP；Ascend还存在另一条MoE执行路径 | 分别核对真实后端，补对应shape、通信与参数布局，不能直接沿用DeepEP身份 |
| PP | SGLang模型有make_pp_layers及阶段入口/出口；GPU默认搜索并不自动扫PP | 核对GLM共享indexer在切层边界的条件、分层内存、传输与调度模型 |
| CP/DCP | GPU模型类把DSA DCP限定为vLLM；当前SGLang-NPU源码实际已有DSA DCP专用路径 | 以NPU实际实现为准补采与建模，不能照搬GPU旧限制或普通注意力计时 |

运行时基线仍为SGLang提交 `baa9befa381511cd4d56f83980951e5a45fffc6d`。
源码只证明实现路径存在；需要进一步执行验证后才能标注支持。
实际运行时的源码参考：

- `sglang/srt/models/glm4_moe.py`：GLM DSA继承DeepseekV2实现。
- `sglang/srt/models/deepseek_v2.py`：模型分层、PP及MoE执行。
- `sglang/srt/arg_groups/overrides.py`：DeepEP的EP调整、PP关闭overlap。
- `sglang/srt/hardware_backend/npu/attention/dsa_dcp.py`：返回LSE的NPU DCP稀疏注意力。
- `sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`：DCP的Q聚合与输出合并。

来源为上述固定提交的运行镜像，本次在临时目录读取源码，没有把其他实现复制进模拟器。

## 阶段记录（由新到旧）

### 2026-10-25：并行搜索交付与通用workload的实际边界

8K/1K的1800个候选全部完成：132个合并部署、1668个PD，0失败。
更新后的运行镜像包含20种结构声明、完整当前算子表、38种DCP配置的预测支持；
GPU原有两组golden回归390项通过，新增与已有Ascend Rust检查42项通过。
本轮数据阶段提交 `c01901e9`，prefill batch插值提交 `1c91c5dd`。

输入输出不是固定8K/1K。TP8/ADP2/EP16下，两种部署均通过1K/128、2K/512、4K/2K、
8K/4K、12K/1K。安装版4K/2K试搜16个候选，15个可行，1个仍因shape缺少采样边界而失败。
新加入的eager prefill batch插值使用相邻实测batch，保留每个端点的序列/prefix边界与graph bucket身份。
它没有提供范围外外推；不能将上述结果称为任意workload与GPU完全等价。

GPU DSA的解析外推与910C当前的有界查表存在差异。16K/1K、32K/1K会遇到未采集prefix，
极短decode也会越过实测下界。下一项功能工作是通用shape覆盖/外推策略，随后再进行服务准确性校准。
具体命令、已验证负载和失败边界见 [使用文档](910c-glm52-w8a8-usage.md)。
完整证据在 `production/automatic-search-1800-20261025/` 和 `production/parallel-delivery-20261025/`。
后者相对于 `benchmarks/evidence/ascend910c-glm52-w8a8/`。

### 2026-10-25：小TP与PD/DCP数据已接入，正在更新交付镜像

默认结构已扩展到20种：两种MoE路径各覆盖attention TP1/2/4/8/16与全局16/32。
8K输入、1K输出、内存比例0.90下，原生内存检查留下16种可行结构；
单节点的TP1/TP2四种结构无法满足当前模型和上下文的内存要求，不会进入自动搜索。
132个合并部署候选已全部完成；包含PD的1800候选联合搜索正在运行。

TP1/2新增计算数据已发布，当前线性2645、向量1194、attention/indexer2577、BMM366条。
PD扩展到50条映射/页布局曲线、227条稳定记录，全部覆盖8192token；
227个实测点逐条通过canonical模型查询。补测按完整曲线选择，不删除样本中的抖动或挑选更快点。

DCP新增304个TP2用例全部通过数值验证，303条计时达标，1条CV超过10%保留但不入库。
合并后2994条DCP记录全部通过原生查询。TP2/ADP16的DeepEP与张量MoE分别完成8条8K/1K请求，
加上此前36种结构，共38种DCP配置通过预测；TP2/ADP8在示例0.86内存比例下被容量检查拒绝。

GPU版当前的recommend同样不搜索CP/DCP，DCP使用显式predict配置；PD与DCP组合仍被固定运行时拒绝。
GPU Replay自身限制PP1，CP2和混合MoE-TP/EP还存在已记录的NPU运行时问题，不能将放宽校验视为适配。
当前仍需完成联合搜索、重新打包统一wheel和用户镜像，并验证安装后的命令。

### 2026-10-25：十二种结构与1296个自动候选

当前源码支持两组并行结构，每组各6种：

| MoE执行方式 | Attention TP | 全局FFN并行宽度 | Attention DP | SGLang参数 |
|---|---|---|---|---|
| DeepEP，MoE-TP1 | 4、8、16 | EP16或EP32 | EP / Attention TP | `--moe-a2a-backend deepep --ep-size 全局宽度` |
| 张量MoE，EP1 | 4、8、16 | MoE-TP16或32 | MoE-TP / Attention TP | `--moe-a2a-backend none --ep-size 1` |

两组都使用 `--tp-size=Attention TP × Attention DP`，新版使用
`--attn-dp-size=Attention DP`。不能直接把模拟器的 `tensor` 数值当成SGLang的全局TP参数。
固定版本的DeepEP和Ascend FuseEP都会把EP调整到全局TP，因此不能用它们部署混合MoE-TP/EP配置。

64device预算、并发8/16/32、合并与PD联合搜索的1296个候选全部完成，失败数为0。
真实权重的MoE-TP16/EP1配合ADP2、ADP4均已完成8K输入/16输出，
常驻模型张量分别为50,480,189,948和52,840,002,044字节，与预测模型一致。
MoE-TP32三种attention布局已通过算子组合及8K/1K合并、PD模拟；两节点完整权重服务误差尚未校准。

Tensor MoE配合ADP时，router、共享专家和专家计算处理全局token，不能沿用DeepEP的本地token分片。
实测prefill每轮有156次全局all-reduce；decode每层走本地reduce-scatter、全局all-gather、
全局reduce-scatter、本地all-gather，已分别接入组合模型。

DCP仍在采集和最终验证。已确认NPU使用128-token页分片，indexer KV保留完整副本；
模拟器的逻辑KV块应设为 `128 × DCP`。新DCP算子包含缓存写入、可选index重映射、Q聚合、
局部稀疏注意力和FP32输出合并，因此组合时去掉旧缓存写入算子，避免重复计算。
最初使用单token分片的实验数据已明确作废，未进入性能数据库；少量数值校验失败的点也不入库。
GPU原有DCP配置与容量检查的46项相关测试通过，NPU内存校验与已有GPU路径分开处理。

### 2026-10-25：七种结构的自动搜索已打通

在原六种DeepEP结构之外，已加入TP16/ADP1/MoE-TP16/EP1。
64device预算、并发8/16/32、合并与PD联合搜索共525个候选，全部完成，失败数为0。
新结构也单独完成8个8K/1K合并请求，以及两个方向各16个混合PD请求；
DeepEP做P、张量MoE做D的验证开启了严格来源检查。

已接入数据：张量MoE126条、线性2183条、向量1034条，逐条原生查询均与表中测量值一致。
线性与向量保留所有原有曲线，只追加新形状。390个GPU基准回归通过。
源码和采集阶段提交为 `654d8936`，新增示例为
`examples/cli/ascend910c-glm52-tensor-moe-8k1k.yaml`。

这里有两种需要明确区分的限制：

- AISimulate当前GPU与NPU的请求回放入口都只接受PP1，见
  `crates/core/src/engine/launch.rs` 的校验；GPU默认搜索也固定PP1。
  真实SGLang能启动某种PP配置，不等于本仓库已经能模拟它的流水调度。
- 当前SGLang-NPU的CP2实际执行缺少方法；PP2与张量MoE组合在传递共享topk时访问
  `OwedOutput.shape` 失败；EP2的普通AscendTPDispatcher在独立探针中出现全局256组计数
  与本地128组权重不匹配。这些都有保存的错误和源码依据，不作为可运行组合开放。

更多attention TP、张量MoE与ADP组合、DCP仍在继续。现有稳定镜像尚未更新，
上述结果来自当前源码和已更新的原生扩展，不能用旧镜像验证这些新功能。

### 张量MoE与CP/DCP的执行验证

六种TP/EP布局及自动搜索已提交为 `98381b91`。调度器修复另通过8个固定配置、32个独立进程、
每次5000请求的Replay逐字节回归，基线为 `2434b021`，没有修改GPU基准答案。

真实模型的TP16/ADP1/EP1使用 `moe-a2a-backend=none`，已完成8K输入/16输出。
它使用AscendTPDispatcher，不能沿用DeepEP的路由和激活耗时。
新采集128个张量MoE案例全部完成数值校验，126个满足CV≤10%，另外2个保留为未入库记录；
4种分片形状的空图反例全部被校验器拒绝。预测算子、参数内存和共享专家数据正在接入。
实测常驻模型张量为49,300,283,900字节，整轮有157次16-rank all-reduce。

DCP2已完成真实8K输入/16输出。首次尝试在图捕获后显存不足，保留了失败记录；
将KV池上限设为32768、静态显存比例设为0.86、prefill chunk设为4096后，同一8K请求跑通。
这是执行路径验证，DCP专用稀疏注意力、LSE合并和KV分片预测仍在适配。

CP2/interleave则在实际服务预热时失败：固定版本的NPU代码调用了不存在的
`DeepseekV2AttentionMLA.rebuild_cp_kv_cache`。源码中只有调用，没有定义。
因此当前版本不能将这条CP路径列为可部署候选；增加算子表不能解决运行时缺失的方法。
完整错误及版本证据保存在 `production/operator-input-state-node32-20261022-cp2-kv32k-20261024/`。

以上扩展尚未打包进现有稳定镜像，整体适配Goal保持进行中。

### 本轮已验证的新范围

源码环境已完成TP4/8/16、EP16/32共6种内部结构的默认枚举，64device预算、并发8/16/32下
351个合并/PD候选全部完成。已修复3P:1D高并发时SGLang调度器把等待KV交接误判为
可以继续执行的问题；修复保留Replay防死循环检查，72项SGLang测试和2042项Rust测试通过。
390项GPU回归通过，GPU基准答案未修改。

新增的数据为DeepEP143条、attention/indexer1707条、PD81条。
PD保留18个完整曲线，每条曲线来自一个独立采集协议；根据覆盖范围选择完整曲线，
不会按最快耗时挑点或把不同协议的样本拼进同一曲线。原始失败与波动记录仍保留。

TP16/ADP1已实测完成8K输入请求及17并发decode输入观察；常驻参数51,292,091,900字节
与模型计算逐字节一致。它的decode使用16/32桶、dense使用6次all-reduce，已经分别建模。
EP32的TP4/ADP8和TP8/ADP4也已通过8K/1K完整请求模拟。

PP2/EP8的真实权重服务也已启动并完成8K输入/16输出请求；这只完成了执行可行性和
边界观察，PP性能模型还未接入。MoE-TP、CP/DCP等工作继续进行，整体Goal保持active。
现有稳定运行镜像仍是上一阶段版本；本轮源码验证与打包发布分开记录，避免旧镜像冒充新功能。

下方为扩展启动时的记录。

自动搜索代码已改为从模型声明的默认结构生成预算内全部副本组合，仍经过既有原生KV容量检查。
初始自动结构保持已验证EP16布局；后续每种新布局验证后再加入。
64device预算下预计为24个合并候选和72个PD候选（各包含并发8/16/32），
共96个。这里增加的是副本和比例覆盖，并不把它冒充为新增96种内部并行方式。
枚举与模型检查已有24项测试通过，完整CLI验证正在进行。

EP32补采使用32和19节点，原始完整88个case计划保留，单次平均从64次调用增加到256次，
仍要求完整载荷校验与CV≤10%。双节点44个case已完整尝试：40个执行完成，
2个已知normal M2048 tiling失败，另2个低延迟case出现向量核/访存异常，均保留原始记录。
均匀路由graph的M1、M3和M32已取得稳定dispatch/combine边界，可为小batch范围提供插值端点。
19节点正在执行同协议的单节点部分；完整计划验证前不会替换正式表。

32节点同时启动当前真实权重的TP16/ADP1状态观察，记录算子输入、边界和内存。
这些插桩结果用于确定执行结构，不作为未插桩的服务性能测量。

后续每完成一个独立阶段继续本地Git提交。本轮Goal保持进行中，未完成整体扩展。
