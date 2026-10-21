# 910C 单机、双节点硬件采集与 YAML 适配

更新日期：2026-10-21。本文承接 [单机采集说明](910c-single-node-collection.md)。

## 1. 这次完成了什么

已经生成当前仓库可识别的系统定义：

```text
python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml
```

它采用现有 `data_dir / gpu / node / misc` 结构，系统名注册为 `ascend910c`。
本次交付的含义是：**有来源、有原始样本、有质量检查的硬件校准配置**。
它还不意味着“GLM-5.2 W8A8 的算子级预测和完整请求预测已经适配完成”。

机器、镜像、采集脚本、逐 rank 数据、失败尝试、样本统计和参数来源保存在：

```text
benchmarks/evidence/ascend910c-glm52-w8a8/
├── hardware-calibration.json
├── onboarding-checkpoint.json
├── single-node/run-20261020-231954/
└── dual-node/
    ├── run-20261021-145053/       # 原始 native 测量，含主机下发间隙
    ├── queued-20261021-151553/    # 最终采用的设备侧通信测量
    └── restored-serving/        # 恢复服务后的 8192/1024 功能验证
```

## 2. 一张 910C 到底对应模拟器几个 device

实机产品名为 `Atlas 800T A3`，运行时设备名为 `Ascend910_9382`。
这两台机器各有 8 块双芯片板卡，运行时分别暴露 **16 个 NPU device**。
本 YAML 中的一个 device 是一个运行时 chip，不是一整块双芯片板卡。

因此：

- `node.num_gpus_per_node = 16`。字段保留项目原有名称，同样用于 NPU。
- 显存容量、HBM 带宽、BF16/INT8 算力都是“每个运行时 chip”的数值。
- TP16 占用单节点的 16 个 device。不能用“8 张卡”把 TP16 错算为两个节点。
- `npu-smi` 每个 chip 显示 65536 MiB；运行时报告的可寻址总容量更小。
  YAML 取 16 个 chip 中最小的运行时总容量 **65,787,658,240 bytes**，
  不是当前空闲显存，也不是模型加载后的 KV 可用容量。

## 3. 两个节点和软件环境

| 项目 | 本机 | 临时节点 |
| --- | --- | --- |
| 主机 | npu1-32 | npu1-19 |
| 控制面地址 | 10.120.72.32 | 10.120.72.19 |
| 可见 NPU chip 数 | 16 | 16 |
| 产品 | Atlas 800T A3 | Atlas 800T A3 |
| SuperPod ID | 1 | 1 |
| Server Index | 31 | 18 |
| 驱动 | 26.1.1 | 25.5.0 |
| HAL | 7.35.23 | 7.35.23 |

两端采集使用相同的不可变镜像：

```text
quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443
```

镜像中为 SGLang 0.5.22、CANN/HCCL 9.1.0 B243、torch 2.10.0、
torch-npu 2.10.0.post6。两端驱动版本不同这一点被保留在证据中。

**跨节点结果只适用于已测的同一 SuperPod 内两节点。**
两端 SuperPod 信息一致，HCCN IP 查询没有普通网口地址；测得跨节点带宽
也远高于主机 bond4 的 25GbE 链路。不能把控制面 bond4 的带宽填进 NPU
数据面 `inter_node_bw`，也不能把本次结果推广到跨 SuperPod 的 RDMA 网络。

## 4. YAML 中每个关键数字从哪里来

以下 GB/s、TFLOP/s、TOPS 使用十进制单位。

| 字段 | 写入值 | 来源与含义 |
| --- | ---: | --- |
| `gpu.mem_capacity` | 65,787,658,240 B | 16 个 chip 的运行时总容量取最小值 |
| `gpu.mem_bw` | 1,269,033,356,802 B/s | 512 MiB `Tensor.copy_`，读写流量合计；各 chip 重复测量后取中位数，再取设备间中位数 |
| `gpu.mem_bw_empirical_scaling_factor` | 1.0 | 带宽已经是实测有效值，不再额外乘 NVIDIA 配置里的 0.8 |
| `gpu.bfloat16_tc_flops` | 321,185,140,687,191 ops/s | BF16 4096³ GEMM，16 个 chip 的有效吞吐中位数，约 321.19 TFLOP/s |
| `gpu.int8_tc_flops` | 553,066,482,547,203 ops/s | INT8 4096³ GEMM，16 个 chip 的有效吞吐中位数，约 553.07 TOPS |
| `node.num_gpus_per_node` | 16 | 运行时 chip 数 |
| `node.intra_node_bw` | 130,673,385,019 B/s | 节点内 8 对跨板卡并发通信，256 MiB 消息，每对单方向有效带宽中位数 |
| `node.inter_node_bw` | 126,790,338,165 B/s | 两节点间 16 对并发通信，256 MiB 消息，每对单方向有效带宽中位数 |
| `node.p2p_latency` | 0.00001328329980375 s | 小消息设备侧 HCCL 点对点耗时，约 13.28 μs，含协议处理 |

**计算字段在这里采用实测校准系数，不能称为厂商理论峰值。**
目前没有核实到可直接用于本机具体 SKU、且单位粒度一致的理论峰值来源，
因此没有猜测 400T、800T 等数字。8192³ 等其他 shape 的有效吞吐也不同，
原始数据保留了这些差别；例如单个 INT8 8192³ case 高于 4096³ 系数。
用这个 YAML 进行 SOL 计算时，结果属于解析校准估计，不具备严格物理下界的保证。

INT8 采集调用 `torch_npu.npu_quant_matmul`，输入已经是 INT8，使用单位
per-channel scale、BF16 输出。它不含输入量化，不等于完整 ModelSlim
W8A8 路径，也不等于融合 MoE、grouped GEMM 或通信量化。

## 5. 通信怎样采集，为什么补测了一次

原生探针基于 ACL、HCCL、MPI 实现，使用 HcclSend/HcclRecv 做 ping-pong：
发送端发送整个 payload，接收端把整个 payload 回传。
每轮的往返时间除以二，得到“平均单方向协议耗时”。

覆盖五种拓扑：

| 组 | rank 数 | 实际路径 |
| --- | ---: | --- |
| sio-2 | 2 | device 0 与 1，同一双芯片板卡 |
| hccs-2 | 2 | device 0 与 2，跨板卡 |
| local-16 | 16 | 节点内 8 对同时通信，0↔8、1↔9 等 |
| cross-2 | 2 | 两节点各一个 chip |
| cross-32 | 32 | 两节点各 16 个 chip，同时形成 16 对跨节点通信 |

每组覆盖 13 个消息大小：64 B 到 256 MiB。
每个 rank、每个大小先预热 50 次，再做 9 轮，每轮 50 次往返。
同步、MPI barrier、数据上传下载都不计入设备事件区间。

第一轮直接在 C++ 下发循环外放设备事件，虽然比 Python 更稳定，
小消息仍会等 CPU 下发：约 60 μs 不代表实际链路延迟。
尝试图捕获时返回 ACL 107025，失败记录保留，没有把失败实验算作数据。

最终采用独立的 host callback gate：先阻住通信 stream，把开始事件、
整轮 HCCL 操作、结束事件全部排好；所有 rank 准备完成后再释放。
设备事件不包含 CPU 排队下发间隙。wall time 单独记录，仍包含这部分时间。
这时节点内跨板卡小消息约 8 μs，跨节点约 13.28 μs。

最终结果保存在 `queued-20261021-151553/`：

- 54 个 rank 实例 × 13 个大小 = **702 个采集点**。
- 每点 9 轮，共 **6,318 个设备计时样本**，另有对应 wall time。
- 所有点 CV 均不超过预先设置的 10%，最大约 **8.25%**。
- 每个 rank、每个大小在计时前后都校验整个 payload，共 1,404 次校验。
- 没有删除慢样本，也没有从不同采集轮次挑快值混合生成曲线。

主机、芯片、peer 映射、样本数、重复编号、完整结束标志和源文件哈希均进入校验。

## 6. HCCL 集合通信交叉检查

使用镜像自带的 HCCL Test 源码在独立容器中编译，记录构建参数和二进制哈希。
分别测节点内 16 rank、两节点 32 rank 的 AllReduce、AllGather、
ReduceScatter、AllToAll，BF16，1 KiB 至 256 MiB。

使用 `-t 1` 设备侧计时、50 次预热、50 次迭代、`-c 1` 结果校验。
AllReduce 重复启动 3 次；其他集合操作是检查性测量，不能称为完整重复性认证。

这里发现一个必须区别处理的问题：BF16 SUM ReduceScatter 的测试输入范围
触发工具的 overflow 检查，工具主动关闭校验，输出 `NULL`。
这 14 个点保留为 **未验证**，不能因为进程返回 0 就算成功。
补测 BF16 MAX ReduceScatter 的 14 个点全部通过，但 MAX 是单独的控制实验，
不替代 SUM 的正确性验证。第一轮其余 70 个集合通信点显示校验成功。

HCCL Test 的 `alg_bandwidth` 是算法数据量除以耗时，不能不加换算就作为
物理链路带宽。因此 YAML 带宽取自独立 P2P 探针，而非 AllReduce 的输出列。

## 7. 准确性边界：数据稳定不等于所有预测都准确

当前 Rust P2P 非 SOL 分支使用：

```text
latency = payload_bytes / inter_node_bw + p2p_latency
```

用本次跨节点的小消息和 256 MiB 点定参数，再检查未参与定参数的大小：

- 1、4、16、64、128 MiB：最大相对误差约 **1.61%**。
- 4、16 KiB：约 4%～5%。
- 64 KiB：约 20%。
- 256 KiB：约 **58%**。

256 KiB 附近存在协议/执行路径变化，单一“带宽＋固定延迟”无法表达。
原始测量有重复性，不代表简化公式在全范围准确。
现有 P2P 分支固定使用 `inter_node_bw`，SOL 分支还会去掉固定延迟；
这里没有偷偷改公式、增加一条独立预测通道或用手工拟合替代现有接口。

后续要准确预测 GLM 的真实通信，仍需按 HCCL 算子、dtype、通信组、消息大小、
融合边界和执行方式接入曲线。节点内 SIO 和跨板卡 HCCS 也不能用一个带宽
完全表达，当前节点内值采用更贴近 TP16/EP16 的跨板卡并发路径。

## 8. 为什么 YAML 有些字段没有填

没有用 B300 的值补空白。以下字段尚无适用测量/定义，明确省略：

- `fp32_flops`：项目要求 scalar/vector FP32，不能用允许 TF32 的矩阵测试代替。
- FP8、FP4 吞吐、PCIe 带宽、TDP、内存操作固定延迟、rack 规模。
- `sm_version`：CUDA SM 编号不适用于 Ascend。
- `nccl_version`、`nccl_mem`：Ascend 使用 HCCL，不能伪装成 NCCL。
- `other_mem`：不能从一次硬件探针内存变化推导实际服务的常驻开销。

`misc: {}` 是诚实的范围声明：当前 NCCL 专用的 legacy 内存估计逻辑不能
直接据此给出可信的 Ascend 部署显存建议。后续应实现 HCCL 的内存与通信语义，
或走有真实初始化内存观测的精确 profile；不能填几个 0 让流程假装通过。

数据目录没有放空壳 Parquet，也没有登记一个没有数据的 SGLang query version。
注册硬件名与发布一个可预测的模型/硬件/backend/version 单元是不同步骤。

### 已执行的软件验证

- 当前 checkout 的原生扩展编译成功；Rust 系统配置测试 **5/5 通过**，包括
  新增的 910C 类型化 YAML 读取、16/32 device 拓扑选择和缺省字段检查。
- Python `TestSupportedSystems` **6/6 通过**，通过实际 `load_system_spec` 加载，
  检查硬件注册、device 粒度、BF16/INT8 选择，以及 FP8/FP4 缺失时明确报错。
  此次定向测试使用 `--noconftest` 排除无关的 CLI/generator 全局 fixture；
  使用本 checkout 刚编译的真实 Rust 扩展，没有 mock 原生模块。
- YAML 的 **9 个关键字段**逐项与校准 manifest 对照；原始样本、冻结源代码、
  构建与环境证据共 **149 项 SHA-256 校验通过**。
- 使用仓库指定的 Ruff 0.14.1 检查生产 Python 修改与测试，lint、format 通过。
- 加载开销检查在模块导入之后计时：首次读取约 1.67 ms，10,000 次缓存读取
  平均约 23.90 μs。它不是整轮 forward 的运行时间；本次没有改变估计器热点路径。

日志在 `dual-node/python-hardware-tests-final.log`、`rust-hardware-tests.log`、
`hardware-validation.json` 和 `lint-pinned.log`。首次缺少原生扩展的失败日志
也保留，不能把当时的失败描述成已通过。

### 采集实现与复现入口

`dual-node/` 中的 `prepare.py` 准备同镜像的独立容器，`build_native.py`
编译镜像自带 HCCL Test 和本项目探针，`setup_mpi.py` 仅在采集容器内配置
临时密钥与端口 22290。前面三个脚本从标准输入接收临时节点密码，不保存凭据。
`run_campaign.py --queued` 运行最终采用的设备队列计时，采集结束自动恢复 United。
`summarize.py <本轮输出目录>` 检查样本完整性、映射、校验结果和 CV。

复现时先把这些脚本复制到新的 campaign 目录，并重新确认两端资源空闲、服务名、
地址和端口；不要覆盖已被 checkpoint 引用的本轮构建日志或原始样本。
实际运行使用的代码版本在每轮 `source/` 中冻结。后续给入口补充的故障清理和
恢复超时检查不改变已采集的数据，也不改写冻结源代码。

## 9. 服务恢复与后续工作

两轮独占采集只暂停了本任务部署的 `glm52-united-v0522`，由脚本 finally 恢复。
临时节点原有 `xq-sglang-glm5.2` 容器保持原状。
两端专用采集容器已删除，其临时 SSH 私钥、授权和 22290 端口一并移除，
没有更改主机的 SSH 授权文件。另一个只用于 CPU 编译验证的容器
`aisim-910c-validation-20261021` 停止后保留编译缓存，方便后续继续适配。
最终恢复后 `/health` 正常，两次短请求及精确 **8192 输入、1024 输出**请求通过。
这次长请求 cache token 为 0、retraction 为 0，单次客户端耗时约 52.78 秒，
首 token 约 2.59 秒；这是恢复功能验证，不能据此声称预测误差达标。

接下来优先做：

1. 核对本地 `glm-q` 的 ModelSlim 精度映射，落实 DSA/indexer、MLA、MoE 的
   真实 shape、实现与融合边界，固定 United 的 TP16、attention DP2、EP16。
2. 让 Collector 在这份固定镜像上采集这些实际算子，特别是 INT8 grouped GEMM、
   动态量化、DSA/indexer、attention、MoE dispatch/combine 和 HCCL SUM。
3. 按当前性能数据库契约接入数据，补 Ascend/HCCL 的精度、通信和内存语义。
4. 先比较纯 prefill、纯 decode 的 forward 分解，再验证 8k/1k 完整请求的
   TTFT、TPOT、吞吐和 KV 使用；PD 分离另做 KV 传输验证。

Onboarding 检查点仍保留配置 `united-tp16-dp2-ep16`，未伪造 profile acceptance。
当前完成的是硬件采集与校准定义；GLM/SGLang 的正式 FPM 采集和测量准确性认证
继续作为独立待办，不能仅凭这个 YAML 跳到“完整适配完成”。
