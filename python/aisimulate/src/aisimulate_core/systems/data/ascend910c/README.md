# Ascend 910C hardware calibration

`ascend910c.yaml` describes one runtime-visible `Ascend910_9382` chip in an
Atlas 800T A3 node. A node exposes **16 chips on eight dual-chip boards**.
Memory capacity and throughput are per chip, never per board or per node.

This directory contains an experimental source-qualified linear table under
`gemm/sglang/0.5.22/ascend_linear_perf.parquet`: 1,241 stable measurements from
1,246 attempted cases, with raw provenance and collection metadata alongside it.
It distinguishes static W8A8, dynamic W8A8, BF16, the indexer weighting projection's
FP32-input/BF16-weight/FP32-output path, bias, storage format and
eager/graph execution. Native queries use exact keys and interpolate only inside
measured token brackets. Five unstable measurements are excluded and retained
in provenance. Missing curves and out-of-bracket queries fail explicitly.

This is **partial operator support**, not a complete GLM prediction database.
The system-specific query-version override pins SGLang to `0.5.22`; the Ascend
runtime belongs to a separate compatibility domain and does not promote the
CUDA systems' fleet-wide `next` version. This pin is not a model coverage claim.
Registering the hardware name or loading this table does not qualify GLM-5.2,
SGLang-Ascend, ModelSlim INT8 MoE,
whole-forward prediction, or replay accuracy. No NVIDIA communication data or
backend version is inherited. The existing NCCL-based memory estimator also
requires a separate HCCL integration; `misc` deliberately has no invented NCCL
version, communication-memory reservation, or serving-memory allowance.

Linear collection uses the pinned image below, driver 26.1.1, 64 independent
weight copies and 128 calls per timing sample. All attempted cases pass sampled
CPU-reference checks through the actual eager invocation or captured replay.
Matched M=8 controls on nodes 31 and 32 differ by at most 4.4% in paired means.
Working-set sensitivity reaches 9.4%. Earlier linear grids failed independent
held-out accuracy checks, including kernel-tiling discontinuities. The current
mixed-precision grid has verified execution and native table lookup, but its
interpolation, whole-forward and serving accuracy remain uncalibrated. Functional
bring-up takes priority over prediction-accuracy tuning. Original observations and intentional graph-noop failures are
under `benchmarks/evidence/ascend910c-glm52-w8a8/production/linear*`.

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
- Historical hardware calibration used local driver 26.1.1 and peer driver
  25.5.0; both report HAL 7.35.23. Revalidate affected measurements after a
  driver/runtime/topology change. Results are not qualified
  for more than the two measured nodes or for loaded/co-tenant hardware.

Unmeasured scalar FP32, FP8, FP4, PCIe bandwidth, TDP, memory fixed latency,
rack size and serving-memory overhead are omitted. CUDA `sm_version` is not
applicable. Missing compute modes retain the existing explicit error behavior;
do not derive their throughput from BF16 or INT8.
