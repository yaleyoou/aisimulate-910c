import datetime
import json
import sys

import torch
import torch_npu


def main():
    result = {
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device_count": torch.npu.device_count(),
        "devices": [],
        "model_loaded": False,
    }
    try:
        for device_id in range(result["device_count"]):
            properties = torch.npu.get_device_properties(device_id)
            result["devices"].append({
                "id": device_id,
                "name": torch.npu.get_device_name(device_id),
                "total_memory_bytes": properties.total_memory,
            })
        torch.npu.set_device(0)
        matrix = torch.ones((16, 16), device="npu:0", dtype=torch.bfloat16)
        product = matrix @ matrix
        torch.npu.synchronize()
        result["bf16_matmul_passed"] = bool(torch.all(product.cpu() == 16))
        result["device_0_free_total_bytes"] = list(torch.npu.mem_get_info(0))
        result["success"] = result["device_count"] == 16 and result["bf16_matmul_passed"]
    except Exception as error:
        result["success"] = False
        result["error"] = repr(error)
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
