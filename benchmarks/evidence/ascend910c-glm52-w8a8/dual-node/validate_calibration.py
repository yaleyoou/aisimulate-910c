import hashlib
import json
import math
import pathlib
import sys
import time

import aisimulate


def main():
    native_path = pathlib.Path("/tmp/aisim-native")
    if native_path.exists():
        aisimulate.__path__.append(str(native_path))
    from aisimulate_core.sdk import common
    from aisimulate_core.sdk.errors import MissingSystemFlopsError
    from aisimulate_core.sdk.perf_database import load_system_spec
    from aisimulate_core.sdk.system_spec import SystemSpec

    evidence = pathlib.Path(__file__).resolve().parents[1]
    manifest = json.loads((evidence / "hardware-calibration.json").read_text())
    started = time.perf_counter()
    spec = SystemSpec(load_system_spec("ascend910c"))
    cold_ms = (time.perf_counter() - started) * 1000
    for name, entry in manifest["fields"].items():
        block, field = name.split(".")
        assert math.isclose(spec[block][field], entry["value"], rel_tol=1e-12, abs_tol=0), name
    for relative, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((evidence / relative).read_bytes()).hexdigest() == expected, relative
    verified_hashes = len(manifest["source_sha256"])
    for run in ["run-20261021-145053", "queued-20261021-151553"]:
        directory = evidence / "dual-node" / run
        summary = json.loads((directory / "summary.json").read_text())
        for relative, expected in summary["source_sha256"].items():
            assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected, relative
            verified_hashes += 1
        campaign = json.loads((directory / "manifest.json").read_text())
        assert campaign["status"] == "collected" and campaign["service_restored"]
        for relative, expected in campaign["source_sha256"].items():
            assert hashlib.sha256((directory / "source" / relative).read_bytes()).hexdigest() == expected, relative
            verified_hashes += 1
    assert "ascend910c" in common.SupportedSystems
    assert spec.get_p2p_bandwidth(16) == spec["node"]["intra_node_bw"]
    assert spec.get_p2p_bandwidth(32) == spec["node"]["inter_node_bw"]
    assert common.get_quant_tc_flops(spec, common.GEMMQuantMode.sq) == spec["gpu"]["int8_tc_flops"]
    rejected = []
    for mode in [common.GEMMQuantMode.fp8, common.GEMMQuantMode.nvfp4]:
        try:
            common.get_quant_tc_flops(spec, mode)
        except MissingSystemFlopsError:
            rejected.append(mode.name)
        else:
            raise AssertionError("Unmeasured precision silently accepted")
    smoke = json.loads((evidence / "dual-node/restored-serving/serving-smoke.json").read_text())
    assert smoke["status"] == "passed"
    workload = next(case for case in smoke["cases"] if case["case"] == "8192_1024")
    assert workload["meta_info"]["prompt_tokens"] == 8192
    assert workload["meta_info"]["completion_tokens"] == 1024
    started = time.perf_counter()
    for _ in range(10000):
        load_system_spec("ascend910c")
    warm_us = (time.perf_counter() - started) * 1e6 / 10000
    print(json.dumps({"status": "passed", "python": sys.version, "verified_hashes": verified_hashes,
                      "yaml_fields_checked": len(manifest["fields"]), "unknown_compute_modes_rejected": rejected,
                      "source_integrity": "verified", "serving_smoke": "8192/1024 passed",
                      "load_after_module_import_ms": cold_ms, "warm_load_microseconds_mean_10000": warm_us,
                      "performance_scope": "Hardware loader only; import/native build time excluded. One static registry entry; no estimator hot-path changes.",
                      "formal_fpm_accepted": False, "serving_prediction_accuracy_qualified": False}, indent=2))


if __name__ == "__main__":
    main()
