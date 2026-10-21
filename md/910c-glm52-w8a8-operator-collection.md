# 910C + GLM-5.2 W8A8 算子采集交付

更新日期：2026-10-22。采集目录中的 `20261021` 是开始日期；独立算子验证在 10 月 22 日完成。

## 当前做到哪里

第 1 项“模型到真实执行的映射”已经完成并提交 `2a0ad787`。
硬件及之前的基线提交是 `93011105`。映射详见同目录
`910c-glm52-w8a8-operator-mapping.md`。

第 2 项已交付可执行的采集器、真实 United 轨迹、独立算子数值与计时验证以及失败记录。
这是固定配置的首轮采集交付；完整性能库和最终预测精度仍须后续验收。

## 两类数据分别能说明什么

| 数据 | 范围与结果 | 适用边界 |
|---|---|---|
| United 实际执行轨迹 | npu1-32，真实权重，TP16 / attention DP2、TP8 / EP16，8k 输入、1k 输出 | 用于确认运行时 kernel、shape、dtype、融合和次数 |
| 独立算子验证 | 空闲 npu1-19，同一固定 SGLang 镜像，36 个 synthetic case | 用于 CPU 数值抽查与无 profiler 的重复计时 |

两台机器驱动版本不同：npu1-32 为 26.1.1，npu1-19 为 25.5.0。
独立探针还使用合成权重、均匀 expert 路由和自定义 attention 数值控制，
因此没有把两类数据混成同一张生产预测表。

### United 轨迹的验收

- 一次无 profiler 对照，加三次独立 profiler 请求；四次都完成 8192 输入 / 1024 输出。
- 全部 `cached_tokens=0`、`num_retractions=0`、固定 DP0；四次输出的 1024 个 token ID 完全一致。
- 每轮覆盖 16 rank、每 rank 3 次 forward：一次 prefill、两次 decode。
- 通过 480 项按 rank、prefill/decode 和算子类型的次数检查。
- 共保留 474,285 条设备记录。按 rank、graph ID、kernel、shape、dtype、layout 分为 15,326 组，
  其中 13,954 组的三轮中位数 CV≤10%，1,372 组不合格并保留原因。
- 不合格组集中在通信及部分短操作。它们不会被补值成“准确耗时”。

无 profiler 对照 TTFT 约 2.538 秒，完整请求约 52.713 秒；profiler 请求完整耗时约 70 秒。
这说明 profiling 存在明显开销，不能拿带 profiler 的请求耗时验证离线预测精度。
这组数值也是功能与采集对照，尚不是完整吞吐 benchmark。

第一次三轮 campaign 的首轮发现几个 rank 的 decode kernel 记录截断。
`analyse.done` 文件虽然存在，次数仍不足。检查程序明确拒收，保留原始失败记录，再完整重采三轮。
最终采用的是 `operators/verified-20261021/`，不能把早期 `repeat-20261021/` 当作验收通过的数据。

### 独立探针的验收

| 类型 | case 数 | 检查方式 |
|---|---:|---|
| 静态 W8A8 linear | 8 | 使用 ModelSlim builder 创建正确 dtype/layout；量化整数输入与 CPU matmul 抽查 |
| 动态 W8A8 linear | 8 | 同一框架实现；逐 token scale 与 CPU matmul 抽查 |
| INT8 grouped matmul | 6 | 每 rank 16 experts、两种投影维度、三种 token 数；每个 expert 抽查 |
| 动态量化 | 6 | CPU absmax/127 scale 和整数 rounding 对照 |
| BF16 LightningIndexer | 4 | 8k prefill，8k/8.5k/9k decode；CPU top-k 集合及因果范围检查 |
| BF16 sparse attention | 4 | 同四种上下文；CPU QK、RoPE、softmax、V 抽查 |

36 个 case 的数值检查全部通过，重复计时 CV 全部≤10%。
矩阵/attention 使用 `abs_error ≤ 0.02 + 0.012×abs(reference)`；indexer top-k overlap≥99%，
且不能有重复、越界、未来 token；动态量化 scale 误差 `rtol=1e-5, atol=1e-7`，整数最多相差 1 个量化格。
这些都是采样检查，没有声称逐元素穷举或模型输出质量已经完全验证。

计时先做 10 次 eager 预热，再做 10 次 graph replay 预热；每个图含 16 次调用，
用 NPU event 计时、除以 16，保留 9 次独立 replay 的样本。
首次实验发现只预热 eager 会让 graph 首次执行污染结果；原始异常样本已保留，修正方法后重新采集。

## 文件和复现入口

源码位于仓库的 `python/aisimulate/collector/sglang/`：

- `ascend_trace.py`：真实服务采集；自动结束有界 profiling，导出证据。
- `ascend_trace_report.py`：归档、shape/精度身份、重复性汇总；不能代替模型次数校验。
- `ascend_quant_probe.py`：独立量化、矩阵、indexer、sparse attention 数值与计时探针。

所有测量和失败证据位于：

`benchmarks/evidence/ascend910c-glm52-w8a8/operators/`

- `verified-20261021/collection-validation.json`：最终真实服务验收和文件哈希。
- `verified-20261021/kernel-groups.json`：全部 shape/dtype/layout 分组、重复性及未通过项。
- `verified-20261021/profile-*/kernel-evidence.tar.gz`：进入 Git 的原始 kernel CSV、profiler 配置和 step span。
- `verified-20261021/profile-*/diagnostics.tar.gz`：profiler 日志和通信分析导出。
- `isolated-20261022/results.json`、`attention.json`：36 个独立 case 的逐点结果。
- `isolated-20261022/probe.py`：实际执行的源码快照；保留之前失败日志及冷 graph 测量。
- `collector-tests.log`：7 项测试通过；采集器代码通过 Ruff lint 和格式检查。

完整 `trace.tar.gz` 较大，保留在本地、由本目录 `.gitignore` 排除。
Git 内保留的紧凑证据足以重算 kernel 汇总；其余文件的 SHA256 用于审计。
不相关的 `md/test.md` 没有加入本次提交。

复现命令和每个探针参数的源码出处见 `python/aisimulate/collector/README.md` 的
“Experimental Ascend serving observations”。不要在有其他流量的服务上清缓存采集。

## 还没有宣称完成的工作

1. 实际 DeepEP 路由分布下的独立 dispatch/combine 校验、稳定通信曲线和 HCCL 查询契约。
2. 将 Ascend shape、量化、graph padding、group-list、有效 KV 长度接入生产性能数据库和 Rust FPE。
3. 非本次固定单请求配置的 batch / 并发 / 并行布局覆盖，整轮 forward 与预测的误差比较。
4. KV 内存、调度与完整 Replay 的 TTFT / TPOT / 吞吐准确率；PD 分离另行验证。

Formal FPM onboarding 的 `united-tp16-dp2-ep16` 仍为阶段 1 / `in_progress`，
尚未接受精确 profile、未运行正式 FPM campaign；不要把本轮实验性采集完成等同于六阶段全部完成。
恢复入口是 `benchmarks/evidence/ascend910c-glm52-w8a8/onboarding-checkpoint.json`。
