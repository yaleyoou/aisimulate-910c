# 910C + GLM-5.2 W8A8：离线预测与配置搜索

使用标准 `predict/recommend --stack engine`：910C实测算子表 → GLM完整forward组合 →
KV容量推导 → 请求调度模拟 → PD传输 → 配置搜索。
8K/1K是示例负载，不是接口限制。**新版通过 `shape_policy: anchored` 支持数值shape外推，主要示例已开启。**
原始实测表不变；范围内沿用原始结果，范围外使用同身份测量锚点和计算/访存工作量模型。
长度、并发仍须满足模型、KV容量和目标运行时限制，不代表任意硬件配置都能执行。
完整服务的forward、TTFT、TPOT误差校准仍按本轮要求延后。

## 1. 在32节点运行

```bash
cd /home/youxingye/youxingye_data/aisimulate-910c

sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-8k1k.yaml \
  --output-dir aisimulate-output/910c-united

sudo bash scripts/run-ascend910c.sh recommend \
  --stack engine \
  --config examples/cli/ascend910c-glm52-auto-search-8k1k.yaml \
  --output-dir aisimulate-output/910c-auto-search
```

第二个命令自动选择并行结构、合并/PD部署、P:D副本数和并发量。
默认预算64device、并发8/16/32、64条8K/1K请求，共1800个候选。
最优只针对当前负载、候选范围和目标，不代表所有配置的全局最优或实测服务排名。
目标是每device输出吞吐，默认没有延迟SLA；选型还需检查TTFT、TPOT。
1800是该8K/1K示例的候选数量；改变负载、预算或并行范围后，应重新检查搜索空间与trial上限。
快速尝试可加 `--set optimizer.max_trials=12`，但只探索部分候选；
也可用 `--set optimizer.algorithm=bayesian`。

结果是 `recommendation.json`、`recommendation.csv`，排序后的完整配置位于
`recommendations/0001.yaml` 等。复算第一名：

```bash
sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config aisimulate-output/910c-auto-search/recommendations/0001.yaml \
  --capture-per-request \
  --output-dir aisimulate-output/910c-best
```

脚本使用本地 `aisimulate-910c:glm52-w8a8` 镜像，禁用网络，只读取config和量化元数据。
不加载完整权重、不启动SGLang、不申请NPU，预测时不要求19/31节点在线。
默认16核CPU、32GiB主机内存，输出归调用sudo的用户所有；可通过
`AISIMULATE_910C_CPUS`、`AISIMULATE_910C_CPUSET`、`AISIMULATE_910C_MEMORY` 调整资源。
模型默认 `/mnt/raid/user/data/models/glm-q`。换目录需同时设置
`AISIMULATE_910C_MODEL` 和 `--set engine.model=新路径`。

## 2. 修改输入输出长度

例如搜索4K输入、2K输出：

```bash
sudo bash scripts/run-ascend910c.sh recommend \
  --stack engine \
  --config examples/cli/ascend910c-glm52-auto-search-8k1k.yaml \
  --set traffic.source.input_tokens=4096 \
  --set traffic.source.output_tokens=2048 \
  --set engine.context_length=6144 \
  --output-dir aisimulate-output/910c-search-4k2k
```

`context_length` 是输入加输出的上下文上限。修改负载时也要修改它，避免保留示例9216而截断输出。
同样的参数可用于predict。输入输出不必等于采集点：模型推导每轮batch的算子shape后插值。
eager prefill会在已测batch之间插值，遇到不齐整的序列网格时通过同身份锚点补齐。
范围外算子结果标记为 `estimated`，所选策略写入保存配置和PD元数据。
需要只接受原测量范围时，加 `--set engine.estimator_config.op_level.ascend.shape_policy=measured`。
未设置该字段的旧配置仍默认 `measured`；新版四个主要示例明确选择 `anchored`。

本轮已经验证合并部署、PD、DCP三种模式各8组负载，共24组：
1/32、65/32、4K/2K、16K/1K、32K/1K、64K/1K、128K/1K、8K/16K，
每组4条请求，输入输出token总数均检查。64K/128K使用全局32的配置满足KV容量；
不能在单节点容量不足时仅增加 `context_length`。

外推把测量延迟分为固定开销和随shape变化的部分，概念上为：

```text
预测耗时 = 固定开销 + (锚点耗时 - 固定开销) × 目标工作量 / 锚点工作量
```

固定开销取对应实测曲线的最小延迟，不随长度成倍放大。
Linear/BMM/MoE依据计算量和权重/激活访存；Indexer考虑完整因果Q×K工作量，
稀疏注意力考虑top2048上限；DCP另计缓存写入和通信；PD按相同rank映射的KV页数增长。
不跨dtype、硬件、内核、MoE布局或graph bucket套用数据。DeepEP已知执行边界继续保留。

下面保留旧 `measured` 策略的边界对照，说明本次补齐的内容。

已用TP8/ADP2/EP16、4K prefill chunk、并发2、每组4条请求验证：

| 输入/输出 | 合并部署 | PD部署 |
|---|---|---|
| 1024 / 128 | 通过 | 通过 |
| 2048 / 512 | 通过 | 通过 |
| 4096 / 2048 | 通过 | 通过 |
| 8192 / 4096 | 通过 | 通过 |
| 12288 / 1024 | 通过 | 通过 |
| 16384 / 1024 | 缺少prefix=12288的数据 | 同样缺少 |
| 32768 / 1024 | 缺少prefix=12288的数据 | 同样缺少 |
| 65 / 32 | decode序列低于实测下界128 | 同样缺少 |

旧版安装包的4K/2K试搜得到15个可行、1个数据覆盖失败；失败配置为
TP16/ADP1、MoE-TP16/EP1、并发32。它需要batch3、每请求2731个prefill token，
而batch4插值端点的采样上界是2048。新增batch插值仍要求两个端点都覆盖序列长度，
`measured` 因此拒绝该候选，`anchored` 现在可以为这些缺失数值坐标估计耗时。
原测量边界为prefill prefix锚点0/8192、decode序列128～16384；batch、head数、图模式和MoE分布还有各自范围。
12K输入分成三个4K chunk时，prefix为0/4096/8192；16K的第四个chunk需要prefix=12288，超出已有表。
输出变长也会逐轮增加decode的KV长度，不能只把1K输出时间乘一个倍数。

PD多数曲线覆盖128～16384token；4→4、8→8碎片页上界为8192，
16→8、16→16碎片页范围1024～8192，16→16连续页下界1024。
因此输入长度、页布局和P/D映射共同决定哪些结果是范围内插值、哪些需要外推。

GPU部分算子同样使用解析计算/访存模型和实测锚点进行范围外外推。
**不需要为每一种输入/输出组合重新测整个模型**。本轮另采集66个验证点和42个后续复核点，
检验NPU执行与外推趋势；这些点保持独立，不作为新测量行偷偷填入原表。
未知算子身份或不支持的graph bucket仍报错，不使用A100数据或固定8K耗时代替。

## 3. 支持的并行结构

| 模式 | Attention TP | 全局FFN宽度 | Attention DP | MoE参数 |
|---|---|---|---|---|
| DeepEP | 1/2/4/8/16 | 16或32 | 全局宽度 / TP | MoE-TP1、EP16/32 |
| 张量MoE | 1/2/4/8/16 | 16或32 | 全局宽度 / TP | MoE-TP16/32、EP1 |

共20种结构声明。8K/1K、内存比例0.90时留下16种；全局16下TP1/TP2四种结构不满足当前模型和上下文容量。
修改上下文或内存比例后会重新检查可行性。

模拟器 `tensor` 是attention组内TP；SGLang的 `--tp-size` 是全局TP，等于
`tensor × attention_data`，`--attn-dp-size` 对应 `attention_data`。
例如United全局TP16、ADP2、EP16：

```yaml
parallelism:
  tensor: 8
  attention_data: 2
  moe_tensor: 1
  moe_expert: 16
  pipeline: 1
  replicas: 1
```

每副本占 `tensor × attention_data × pipeline` 个device，EP不额外相乘。
DeepEP对应 `--moe-a2a-backend deepep`；张量MoE对应 `none --ep-size 1`。
每节点16个运行时device。`max_candidate_gpus`、`used_gpus`、`throughput_per_gpu` 保留原字段名，
选ascend910c后按NPU device计数。P/D独立选结构，并按预算生成1P:3D、3P:1D等副本组合。
超过两节点的模拟依赖同硬件、同软件、同互联条件的数据复用；共享链路争用尚未校准。

## 4. PD与DCP

```bash
sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-pd-8k1k.yaml \
  --capture-per-request \
  --output-dir aisimulate-output/910c-pd

sudo bash scripts/run-ascend910c.sh predict \
  --stack engine \
  --config examples/cli/ascend910c-glm52-dcp2-8k1k.yaml \
  --output-dir aisimulate-output/910c-dcp2
```

PD计时来自真实SGLang AscendKVManager、MemFabric1.2.1/SDMA，包含rank映射和234组分层KV缓冲区搬运。
KV为BF16，每个完整KV副本每token占109824字节；8K共899678208字节。
W8A8不意味着KV、归一化、路由器和通信全部INT8。

`engine.estimator_config.op_level.ascend.kv_transfer_page_layout` 可选 `contiguous` 或 `fragmented`。
`engine.kv_transfer` 使用 `{type: measured, timing_mode: full_prompt}`；
也可用 `destination_missing` 只计目标缺失token。

DCP2/4/8/16必须整除attention TP。latent/RoPE KV分片，indexer KV仍完整保留；
物理页128token，模拟器 `kv_cache.block_size` 必须是 `128 × DCP`。
38种DCP配置已通过8K/1K预测，包括TP2/ADP16的两种MoE路径。
DCP用于显式合并部署predict；本仓库GPU的recommend也不搜索CP/DCP。
固定NPU运行时不支持PD+DCP，该组合仍明确拒绝。

GPU Replay自身限制PP1；NPU CP2存在缺失方法的执行错误；
混合MoE-TP/EP路径存在实际分组维度错误，DeepEP又强制EP等于全局TP。
部分PP2真实部署可启动，并不代表AISimulate已经具备PP请求模拟。
执行证据与源码位置见 [并行扩展进展](910c-glm52-w8a8-parallel-expansion.md)。

## 5. 数据与解释范围

| 数据族 | 已发布记录数 |
|---|---:|
| Linear | 2645 |
| 向量/归一化/缓存操作 | 1194 |
| Attention/Indexer | 2577 |
| MLA BMM | 366 |
| DeepEP专家计算链 | 225 |
| 张量MoE计算链 | 126 |
| DeepEP通信 | 143 |
| HCCL | 449 |
| PD传输 | 227，组成50条曲线 |
| DCP存储/注意力/合并链 | 2994 |

记录数不是独立内核数；MoE链中8条是已验证的数学等价布局别名。
失败和CV超限记录保留在evidence中，未混入性能表。
发布身份绑定硬件、软件、shape、dtype、计时协议和源码；节点数据复用必须满足相同条件。

Rust模型按shape查表，组合78层：3层dense、75层MoE，21层完整indexer、57层复用。
调度器依据chunk、并发和KV容量决定batch，prefill后逐轮生成输出；
PD通过目标KV预留、传输和源端释放衔接decode，再汇总TTFT、TPOT、吞吐和排队。

算子数值正确、表读取正确、请求完成属于功能证据，完整服务误差尚未校准。
当前使用均衡attention-DP和专家路由，PD bootstrap、逐层流式传输和多请求争用尚未校准。
功耗数据未采集，能耗输出明确不可用。
基线固定为SGLang0.5.22（提交 `baa9befa381511cd4d56f83980951e5a45fffc6d`）、
CANN/HCCL9.1.0.B243、驱动26.1.1。升级需核对执行边界并采集变化部分。

## 6. 文件与重建

- 硬件：`python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml`
- 数据：`python/aisimulate/src/aisimulate_core/systems/data/ascend910c/`
- 组合模型：`python/aisimulate/src/aisimulate_core/sdk/models/ascend_glm52.py`
- 原始证据：`benchmarks/evidence/ascend910c-glm52-w8a8/production/`
- 阶段历史：[生产适配进展](910c-glm52-w8a8-production-progress.md)

`build/ascend910c-runtime/` 为Git忽略的构建产物，含统一wheel、原生库、SHA256身份与离线依赖。
保留它可重建镜像：`sudo bash scripts/build-ascend910c-runtime.sh`。
修改Python、Rust或性能表后必须重建wheel和镜像，挂载源码不会替换已安装的包。

在与基础镜像匹配的ARM64/Linux、Python3.11～3.13和支持edition2024的Rust环境中，
安装 `maturin==1.15.0`，在 `python/aisimulate` 目录执行：

```bash
python3 -m maturin build --release --auditwheel skip \
  --out ../../build/ascend910c-runtime
python3 -m pip download --dest ../../build/ascend910c-runtime \
  ../../build/ascend910c-runtime/aisimulate-*.whl
```

提取wheel中的 `aisimulate/_runtime.abi3.so`，更新 `native-identity.json` 的源码提交、库与wheel的SHA256、
release/extension-module配置，再执行构建脚本。延迟SLA与搜索字段见 `docs/sweeper/`。
