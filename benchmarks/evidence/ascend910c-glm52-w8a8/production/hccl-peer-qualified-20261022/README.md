# Control-data qualification only

These 292 rows passed numerical and repeatability checks and were used to test
the native HCCL reader against actual measurements. They are not serving-profile
calibration for United: this control used task queue1 and default HCCL buffers,
while the actual United launcher uses task queue0 and HCCL_BUFFSIZE650.

The production exporter now requires raw schema v2 with the actual environment
and SGLang process-group options. The older control exporter is retained here
for reproducibility. Do not install this directory as the United production
communication profile. Serving-matched collection runs separately.
