# 910C + GLM-5.2 W8A8：离线预测与配置搜索

当前已经打通：910C实测算子表 → GLM完整forward组合 → KV容量推导 → 请求调度模拟 →
PD实测传输 → 并行策略、P:D副本数、并发量搜索。使用标准 `predict/recommend --stack engine`。
本版优先实现功能；完整服务的TTFT、TPOT误差校准留到下一阶段。

## 1. 在32节点直接运行

```bash
cd /home/youxingye/youxingye_data/aisimulate-910c

sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-8k1k.yaml \
  --output-dir aisimulate-output/910c-united

sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-pd-8k1k.yaml \
  --capture-per-request \
  --output-dir aisimulate-output/910c-pd

sudo bash scripts/run-ascend910c.sh recommend \
  --stack engine \
  --config examples/cli/ascend910c-glm52-search-8k1k.yaml \
  --output-dir aisimulate-output/910c-search
```

脚本使用本地 `aisimulate-910c:glm52-w8a8` 镜像，运行时禁用网络，模型目录只读挂载。
不启动SGLang服务，不加载完整权重，不申请NPU，不要求19/31节点在线。
只读取当前模型的config和量化元数据，以及仓库内的硬件/性能数据。
输出归运行sudo前的用户所有。默认最多16核CPU和32GiB主机内存，可用
`AISIMULATE_910C_CPUS`、`AISIMULATE_910C_MEMORY` 调整；
默认绑定当前可用的前16个CPU，`AISIMULATE_910C_CPUSET` 可指定CPU列表。

模型默认是 `/mnt/raid/user/data/models/glm-q`。若更换目录，需要同时设置
`AISIMULATE_910C_MODEL` 挂载目录，并通过 `--set engine.model=新路径` 修改模型身份。
模型结构、量化布局变化会重新验证；不能仅替换成一个模型名字。

预测结果在 `prediction.json`；搜索结果在 `recommendation.json`、`recommendation.csv`，
排好序的具体配置在 `recommendations/0001.yaml` 等文件。
随机搜索示例会遍历这60个有限候选；也可以使用贝叶斯搜索，在recommend命令后添加
`--set optimizer.algorithm=bayesian --set optimizer.max_trials=12`。
最终环境已验证12次建议、11个实际评估全部成功、1次缓存复用；该次数不保证遍历所有候选。
复算第一名：

```bash
sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config aisimulate-output/910c-search/recommendations/0001.yaml \
  --capture-per-request \
  --output-dir aisimulate-output/910c-best
```

## 2. 示例的含义

| 示例 | 部署与搜索范围 |
|---|---|
| `ascend910c-glm52-8k1k.yaml` | 单个United副本：attention TP8、ADP2、EP16，8条请求、并发2 |
| `ascend910c-glm52-pd-8k1k.yaml` | P1:D1，各16device，attention TP8/ADP2，16条请求、并发4 |
| `ascend910c-glm52-pd-search-8k1k.yaml` | 仅PD，共48候选；P/D独立选择TP4或8、副本1或2，并发8/16/32 |
| `ascend910c-glm52-search-8k1k.yaml` | 合并部署与PD一起搜索，共60候选，目标为每device输出吞吐 |

这里的 `tensor` 是**每个attention-DP组内的TP**。United的全局TP16对应：

```yaml
parallelism:
  tensor: 8
  attention_data: 2
  moe_tensor: 1
  moe_expert: 16
  pipeline: 1
  replicas: 1
```

每个副本占 `tensor × attention_data × pipeline = 16` 个运行时device。
EP描述同一组device上的专家切分，不额外乘一次16。
`P1:D2` 因而需要48个device，也就是3个这样的节点。
项目的 `max_candidate_gpus`、`used_gpus`、`throughput_per_gpu` 字段保留原名称；
选择 `ascend910c` 时其计数单位是运行时NPU device。硬件YAML配置每节点16device。

64device预算允许模拟多副本部署，虽然本次实测使用两台节点。
这依赖相同硬件、软件、拓扑和资源条件的复用假设；多副本共享链路争用尚未实测校准。

搜索示例使用**显式已采集候选**，不要直接删除preset切换为无限制的默认GPU拓扑枚举。
当前可用模型组合为attention TP4/8、EP16/32、MoE-TP1、PP1/CP1/DCP1；
其中EP32仍缺部分小batch DeepEP稳定计时，因此交付的请求搜索限定EP16。
PD传输进一步限定每个P/D副本为单节点16device。

## 3. 8K输入、1K输出实际怎样计算

1. 读取GLM结构和 `quant_model_description.json`，验证78层、3层dense+75层MoE、
   256专家、top8、MLA/DSA及indexer共享模式。只有21层执行完整indexer选择，其余复用。
2. 依据attention TP、ADP、EP和batch，推导各矩阵、向量、注意力、MoE、集合通信的实际shape。
   静态W8A8线性、动态W8A8专家MLP、BF16/FP32操作分别选择对应实测表。
3. Rust原生性能模型按精确键查询；在已测范围内插值。然后按层数、通信顺序和共享专家重叠规则
   组合出本轮forward耗时。不会拿A100数据替代910C缺失点。
4. 请求模拟器按max_batched_tokens、max_sequences、KV容量和队列决定每轮batch。
   8K输入可以一次prefill，也可以分成两个4K chunk；之后逐轮生成1K输出。
5. PD模式在源端完成prefill、目标端预留KV后，调用同一对canonical模型选择KV传输曲线；
   再继续decode。源端释放还需满足已有forward与交接生命周期约束。
6. 汇总每条请求的等待、首token和后续token时间，生成TTFT、TPOT和吞吐。
   recommend对各候选重复这个流程，按指定目标排序，并保存可重新运行的YAML。

KV使用BF16，128token一页。78层MLA主缓存、RoPE部分和indexer缓存合计
每token、每个完整KV副本109824字节；8192token为899678208字节。
W8A8不意味着KV、注意力、归一化、路由器或通信全部INT8。

实测8K PD传输时间：

| P的attention TP → D的attention TP | 连续页 | 碎片页 |
|---|---:|---:|
| 8 → 8 | 7.7815ms | 66.3320ms |
| 4 → 8 | 15.5202ms | 130.4192ms |
| 8 → 4 | 7.7453ms | 65.3545ms |
| 4 → 4 | 7.7941ms | 65.1291ms |

这是实际SGLang AscendKVManager、MemFabric1.2.1、SDMA通路的测量，包含真实rank映射和
234组分层缓冲区搬运。并非用HCCL带宽简单替代。

控制页布局：

```yaml
engine:
  estimator_config:
    op_level:
      ascend:
        routing: balanced
        shared_expert_overlap: true
        kv_transfer_page_layout: contiguous
  kv_transfer:
    type: measured
    timing_mode: full_prompt
```

将 `contiguous` 改成 `fragmented` 可评估碎片页的传输影响。
`destination_missing` 可以只计目标端缺失的token；示例关闭prefix cache并使用full_prompt，
便于明确比较。超出实测范围、不同的模型布局或未采集拓扑会显式报错。

搜索的prefill chunk固定4096，是因为TP4下8K单chunk会使每rank DeepEP normal输入达到2048，
该配置本次在真实内核tiling失败；4K chunk对应1024，已有有效数据。
这不是忽略错误，而是交付一组实际可查询的范围。后续扩展需要补实测，不能放宽范围检查。

## 4. 已验证的结果与使用边界

PD48个候选、联合搜索60个候选全部完成，没有失败候选。
PD搜索在该集合内每device吞吐最优为P1:D2：P为TP4/ADP4，D为TP8/ADP2，并发32，
共48device，预测输出吞吐约166.47token/s，每device约3.468token/s。
保存的候选重新predict得到相同吞吐和TPOT。
联合搜索的最优为合并部署TP4/ADP4/EP16、单副本、并发16，
预测每device吞吐约5.306token/s。该目标没有配置延迟SLA，因此不能只凭排名决定上线参数。

两项排名只针对示例的64条8K/1K请求、所列并行配置、并发量和目标，
不代表所有负载下的全局最优。延迟SLA可通过标准 `evaluation.sla` 配置，详见
`docs/sweeper/optimization-goals.md` 与 `docs/sweeper/search-space.md`。

当前数据包含线性1378、attention/indexer1517、MLA BMM220、向量842、
MoE链225、DeepEP125、HCCL449、PD32条稳定记录。
MoE链中有8条是单token数学等价布局别名；不是额外测量。
失败、波动超限记录保留在evidence中，没有混入稳定计时表。

尚未声明完成的准确性工作：

- 与完整SGLang服务的forward/TTFT/TPOT/吞吐对齐；目前是功能和数据执行正确性验证。
- attention-DP负载不均、真实专家路由偏斜、多并行布局的独立静态内存校准。
- PD多请求链路争用、bootstrap、逐层流式传输及其计算重叠。
- EP32的小batch通信缺口，以及更大shape、PP/CP/DCP/MTP等新策略。
- 功耗预测：尚未采集对应能耗，输出明确显示不可用。

算子级预测使用固定SGLang0.5.22、CANN/HCCL9.1.0.B243、驱动26.1.1基线。
升级软件版本后需要核对融合边界并重新采集变化部分；不能只改backend_version字符串。

## 5. 文件位置与镜像重建

- 硬件：`python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml`
- 性能表：`python/aisimulate/src/aisimulate_core/systems/data/ascend910c/`
- 模型组合：`python/aisimulate/src/aisimulate_core/sdk/models/ascend_glm52.py`
- 采集和验证证据：`benchmarks/evidence/ascend910c-glm52-w8a8/production/`
- 当前阶段进度：`md/910c-glm52-w8a8-production-progress.md`

运行镜像和原生库已经留在32节点。`build/ascend910c-runtime/` 是被Git忽略的本地构建产物，
含统一安装wheel、`_runtime.abi3.so` 和带SHA256的 `native-identity.json`；不是新一份采集数据。
保留该目录即可重建镜像：

```bash
sudo bash scripts/build-ascend910c-runtime.sh
```

在其他ARM64节点重建原生库时，需要Python3.11～3.13和支持edition2024的Rust，
先安装 `maturin==1.15.0`，在 `python/aisimulate` 目录执行：

```bash
python3 -m maturin build --release --auditwheel skip \
  --out ../../build/ascend910c-runtime
python3 -m pip download --dest ../../build/ascend910c-runtime \
  ../../build/ascend910c-runtime/aisimulate-*.whl
```

这会同时打包Python代码、Rust扩展和性能数据。将wheel中的
`aisimulate/_runtime.abi3.so` 提取到构建目录，记录对应Git提交、库和wheel的SHA256、
release profile和extension-module feature到identity，再构建镜像。
当前目录还保留了33个依赖wheel及其PyPI下载SHA256清单；构建时校验清单后使用本地wheel安装。
本地wheel面向与当前固定基础镜像相同的ARM64/Linux环境。
镜像运行使用已安装的统一wheel；挂载仓库用于读取配置与写入输出。
修改实现或性能数据后，应重建wheel和镜像。
