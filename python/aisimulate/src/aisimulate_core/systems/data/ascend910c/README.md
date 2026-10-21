# Ascend 910C hardware calibration

`ascend910c.yaml` describes one runtime-visible `Ascend910_9382` chip in an
Atlas 800T A3 node. A node exposes **16 chips on eight dual-chip boards**.
Memory capacity and throughput are per chip, never per board or per node.

This directory contains **no published operation database**. Registering the
hardware name does not qualify GLM-5.2, SGLang-Ascend, ModelSlim INT8 MoE,
whole-forward prediction, or replay accuracy. No NVIDIA communication data or
backend version is inherited. The existing NCCL-based memory estimator also
requires a separate HCCL integration; `misc` deliberately has no invented NCCL
version, communication-memory reservation, or serving-memory allowance.

The retained calibration manifest is
`benchmarks/evidence/ascend910c-glm52-w8a8/hardware-calibration.json` in the
source checkout. It identifies raw samples, hashes, units, aggregation,
timing boundaries, runtime image, and qualification limits. Measurements were
made on 2026-10-20 and 2026-10-21. Runtime image:
`quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443`.

- HBM bandwidth counts both reads and writes for a 512 MiB copy. The bandwidth
  is already measured, so the additional empirical multiplier is 1.0.
- BF16 and INT8 compute entries are effective throughput of measured
  4096 × 4096 × 4096 matrix operations, aggregated across all 16 chips. They
  occupy the existing compute-coefficient fields but are **not theoretical
  peaks**. SOL using these entries is an analytical calibration, not a proven
  physical lower bound. INT8 input quantization and fused MoE are not included.
- Intra-node bandwidth uses eight concurrent cross-board HCCL ping-pong pairs;
  inter-node bandwidth uses 16 concurrent pairs between `npu1-32` and `npu1-19`.
  Both report SuperPod ID 1. Inter-node numbers apply to this observed fabric,
  not Ethernet/RDMA links between arbitrary nodes or different SuperPods.
- Bandwidth uses the median per-pair throughput at 256 MiB, in bytes/second,
  with each full-payload round trip divided by two. P2P latency uses small
  messages with native calls queued before device execution. It includes HCCL
  protocol work, not CPU submission gaps or solely wire propagation.
- A bandwidth plus constant model cannot reproduce protocol transitions:
  held-out cross-node messages of 1–128 MiB have at most 1.61% error in this
  campaign, but the 256 KiB point has about 58% error. Use collected HCCL
  operation curves before claiming accurate small-message serving predictions.
- Local driver 26.1.1 and peer driver 25.5.0 differ; both report HAL 7.35.23.
  Recollect after a driver/runtime/topology change. Results are not qualified
  for more than the two measured nodes or for loaded/co-tenant hardware.

Unmeasured scalar FP32, FP8, FP4, PCIe bandwidth, TDP, memory fixed latency,
rack size and serving-memory overhead are omitted. CUDA `sm_version` is not
applicable. Missing compute modes retain the existing explicit error behavior;
do not derive their throughput from BF16 or INT8.
