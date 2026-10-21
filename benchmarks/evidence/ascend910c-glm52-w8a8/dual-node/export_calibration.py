import datetime
import hashlib
import json
import pathlib
import statistics


def main():
    evidence = pathlib.Path(__file__).resolve().parents[1]
    local_path = evidence / "single-node/run-20261020-231954/single.json"
    queue_path = evidence / "dual-node/queued-20261021-151553/summary.json"
    local = json.loads(local_path.read_text())
    comm = json.loads(queue_path.read_text())
    baseline = json.loads((evidence / "single-node/hardware-measured.json").read_text())
    runtime = json.loads((evidence / "stage1/runtime-v0522/runtime-image-inventory.json").read_text())
    assert not local["errors"]
    assert len(local["cases"]) == 81
    assert all(case["timing"]["device"]["cv"] <= 0.10 for case in local["cases"])
    assert not comm["cv_failures"] and not comm["collective_unverified_points"]
    assert comm["p2p_samples"] == 6318 and comm["p2p_rank_size_points"] == 702
    effective = baseline["measured_effective_per_device"]
    selections = {
        "hbm_copy_read_plus_write_bytes_per_second": [case for case in local["cases"]
            if case["operation"] == "hbm_copy" and case["size_bytes"] == 512 * 1024 * 1024],
        **{precision + "_4096_cube_ops_per_second": [case for case in local["cases"]
            if case["operation"] == "matmul" and case["shape_mkn"] == [4096, 4096, 4096]
            and case["precision"] == precision] for precision in ["bf16", "int8"]},
    }
    for name, cases in selections.items():
        assert len(cases) == 16 and {case["device"] for case in cases} == set(range(16))
        metric = "effective_bytes_per_second" if name.startswith("hbm") else "effective_ops_per_second"
        observed = statistics.median(case[metric] for case in cases)
        assert observed == effective[name]["median_across_devices"]
    fields = {
        "gpu.mem_capacity": {"value": min(device["total_memory_bytes"] for device in local["devices"]),
                             "unit": "bytes per runtime chip", "source": str(local_path.relative_to(evidence)),
                             "method": "Minimum acl/torch runtime total memory over 16 chips; not free memory and not 128GiB per board."},
        "gpu.mem_bw": {"value": round(effective["hbm_copy_read_plus_write_bytes_per_second"]["median_across_devices"]),
                       "unit": "bytes/second per runtime chip", "source": str(local_path.relative_to(evidence)),
                       "method": "Median across all 16 devices of median repeated 512MiB Torch copy throughput, counting read plus write."},
        "gpu.mem_bw_empirical_scaling_factor": {"value": 1.0, "unit": "dimensionless", "method": "Measured bandwidth already includes utilization; avoid a second arbitrary discount."},
        "gpu.bfloat16_tc_flops": {"value": round(effective["bf16_4096_cube_ops_per_second"]["median_across_devices"]),
                                 "unit": "arithmetic operations/second per runtime chip", "source": str(local_path.relative_to(evidence)),
                                 "method": "Median over 16 chips; 2*M*N*K, M=N=K=4096, torch.mm BF16; measured effective coefficient, not theoretical peak."},
        "gpu.int8_tc_flops": {"value": round(effective["int8_4096_cube_ops_per_second"]["median_across_devices"]),
                             "unit": "arithmetic operations/second per runtime chip", "source": str(local_path.relative_to(evidence)),
                             "method": "Median over 16 chips; 4096-cube torch_npu.npu_quant_matmul, INT8 inputs, unit scale and BF16 output; activation quantization excluded; not theoretical peak."},
        "node.num_gpus_per_node": {"value": 16, "unit": "runtime-visible NPU chips", "source": "dual-node/prelaunch-remote.json"},
    }
    for name, value in comm["system_calibration"].items():
        fields["node." + name] = {"value": value, "unit": "seconds" if name == "p2p_latency" else "bytes/second per runtime chip, single direction",
                                  "source": str(queue_path.relative_to(evidence)), "method": comm["selection"]}
    manifest = {
        "schema": "aisimulate-ascend910c-hardware-calibration/v1", "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "system": "ascend910c", "hardware_definition_status": "measured hardware calibration; not qualified GLM operator or full serving prediction support",
        "system_yaml": "python/aisimulate/src/aisimulate_core/systems/ascend910c.yaml",
        "device_granularity": "one Ascend910_9382 runtime chip; eight two-chip boards / sixteen chips per Atlas800TA3 node",
        "topology_scope": {"nodes": ["npu1-32", "npu1-19"], "superpod_id": 1, "maximum_measured_nodes": 2,
                           "local_server_index": 31, "peer_server_index": 18, "intra_pairs": 8, "inter_pairs": 16,
                           "host_drivers": {"npu1-32": "26.1.1", "npu1-19": "25.5.0"}, "hal": "7.35.23",
                           "transport_evidence": "Same SuperPod metadata; HCCN IP unset; measured cross-host bandwidth exceeds control-plane bond4. Do not infer a general Ethernet/RDMA or cross-SuperPod rate."},
        "runtime_inventory": "stage1/runtime-v0522/runtime-image-inventory.json", "runtime_inventory_sha256": hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest(),
        "runtime_image": baseline["source_runtime_image"], "fields": fields,
        "qualification": {"local_primitive_points": len(local["cases"]), "p2p_samples": comm["p2p_samples"],
                          "p2p_points": comm["p2p_rank_size_points"], "max_repeat_cv": max(point["device_ms_per_direction"]["cv"] for group in comm["groups"].values() for point in group["points"]),
                          "cv_threshold": 0.10, "p2p_payload_correctness": "702 full payload checks before and 702 after timed repeats",
                          "collectives_initial_verified": 70, "collectives_initial_unverified": 14, "reducescatter_max_control_verified": 14,
                          "reducescatter_sum_status": "Unqualified vendor BF16 SUM: tool disables correctness due to its input range. MAX controls are distinct and do not validate SUM.",
                          "graph_capture_status": "Native P2P capture readiness returned ACL107025, retained as failed experiment; excluded from calibration.",
                          "timing": comm["timing_scope"], "held_out_inter_node_bulk_max_relative_error": comm["canonical_inter_node_bulk_held_out_max_error"],
                          "held_out_inter_node_all_sizes_max_relative_error": comm["canonical_inter_node_all_held_out_max_error"],
                          "prediction_accuracy_qualified": False, "formal_fpm_profile_accepted": False, "replay_qualified": False},
        "unmeasured_omitted_fields": ["gpu.fp32_flops", "gpu.fp8_tc_flops", "gpu.fp4_tc_flops", "gpu.power", "gpu.mem_empirical_constant_latency", "node.pcie_bw", "node.num_gpus_per_rack", "node.inter_rack_bw", "misc.other_mem"],
        "inapplicable_omitted_fields": ["gpu.sm_version", "misc.nccl_version", "misc.nccl_mem"],
        "limits": ["Effective 4096-cube compute coefficients are not theoretical peaks; do not claim a strict SOL lower bound.",
                   "A scalar P2P formula misses protocol transitions: about58% error at256KiB; use native HCCL curves for accurate operator predictions.",
                   "No HCCL operator parquet/version mapping or HCCL serving-memory model is published.",
                   "No GLM-5.2 ModelSlim fused-kernel or end-to-end prediction accuracy qualification."]}
    paths = [local_path, queue_path, evidence / "single-node/hardware-measured.json",
             evidence / "dual-node/queued-20261021-151553/manifest.json", evidence / "dual-node/run-20261021-145053/summary.json",
             evidence / "dual-node/local-product.txt", evidence / "dual-node/local_spod.txt", evidence / "dual-node/npu1-19-network.json",
             evidence / "dual-node/native-build-results.json", evidence / "dual-node/npu1-19-native-build.log",
             evidence / "dual-node/npu1-32-native-build.log", evidence / "stage1/runtime-v0522/runtime-image-inventory.json"]
    manifest["source_sha256"] = {str(path.relative_to(evidence)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    target = evidence / "hardware-calibration.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(target), "fields": {name: entry["value"] for name, entry in fields.items()}}))


if __name__ == "__main__":
    main()
