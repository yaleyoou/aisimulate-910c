"""Prepare the authorized peer recording service with verified model content."""

import base64
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dual-node"))
from remote import run_python


ROOT = pathlib.Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[3]


def main():
    password = input()
    verified = json.loads((ROOT / "model-transfer-verified.json").read_text())
    if verified["status"] != "verified":
        raise ValueError("Model content must be verified before launching")
    setup = json.loads((ROOT / "serving-setup.json").read_text())
    files = {}
    for name in ["ascend_state_capture.py", "ascend_recording_server.py"]:
        path = REPOSITORY / "python/aisimulate/collector/sglang" / name
        files[name] = base64.b64encode(path.read_bytes()).decode()
    specification = {
        "name": "aisim-glm52-state-20261022",
        "image": "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443",
        "checkpoint_sha256": verified["checkpoint_sha256"],
        "model": "/mnt/raid/aisimulate-youxingye-20261022/glm-q",
        "output": "/mnt/raid/aisimulate-youxingye-20261022/recording",
        "environment": setup["environment"],
        "files": files,
        "arguments": [
            "--model-path", "/models/glm-q", "--host", "0.0.0.0", "--port", "18000",
            "--context-length", "180000", "--nnodes", "1", "--node-rank", "0", "--tp-size", "16",
            "--dp-size", "2", "--ep-size", "16", "--mem-fraction-static", "0.90",
            "--max-running-requests", "64", "--attention-backend", "ascend", "--device", "npu",
            "--quantization", "modelslim", "--dtype", "bfloat16", "--served-model-name", "glm-5-aisim",
            "--chunked-prefill-size", "16384", "--max-prefill-tokens", "32768",
            "--disable-shared-experts-fusion", "--enable-dp-attention", "--enable-dp-lm-head",
            "--moe-a2a-backend", "deepep", "--deepep-mode", "auto", "--enable-cache-report",
            "--load-balance-method", "round_robin", "--trust-remote-code",
            "--model-loader-extra-config", '{"enable_multithread_load":false}', "--weight-loader-disable-mmap",
            "--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", "32",
            "--cuda-graph-bs-decode", "8", "16", "24", "32",
        ],
    }
    payload = base64.b64encode(json.dumps(specification).encode()).decode()
    remote_source = """
import pathlib,subprocess,json,base64,socket
spec=json.loads(base64.b64decode(PAYLOAD))
if subprocess.run(['docker','inspect',spec['name']],capture_output=True).returncode==0:
    raise RuntimeError('Task recording container already exists; inspect it first')
fingerprint=json.loads(pathlib.Path('/tmp/aisim-910c-production-20261022/model-target-fingerprint.json').read_text())
if fingerprint['checkpoint_sha256']!=spec['checkpoint_sha256']:
    raise RuntimeError('Unexpected target checkpoint identity')
health=subprocess.check_output(['npu-smi','info'],text=True)
if health.count('No running processes found in NPU')!=8:
    raise RuntimeError('NPU process inventory is not empty; preserve existing workloads')
with socket.socket() as probe:
    probe.bind(('0.0.0.0',18000))
root=pathlib.Path(spec['output']);root.mkdir(mode=0o700,exist_ok=False)
(root/'source').mkdir();(root/'capture').mkdir()
(root/'npu-before.txt').write_text(health)
for name,value in spec['files'].items():(root/'source'/name).write_bytes(base64.b64decode(value))
command=['docker','run','-d','--name',spec['name'],'--network','host','--privileged','--shm-size','32g',
    '--mount','type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly',
    '--mount','type=bind,src='+spec['model']+',dst=/models/glm-q,readonly',
    '--mount','type=bind,src='+str(root)+',dst=/campaign',
    '-e','AISIM_CAPTURE_DIR=/campaign/capture','-e','HCCL_IF_BASE_PORT=39000',
    '-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','--entrypoint','python3']
for name,value in spec['environment'].items():
    if value is not None:command+=['-e',name+'='+value]
for path in ['/etc/ascend_install.info','/usr/local/Ascend/firmware','/usr/local/sbin']:
    if pathlib.Path(path).exists():command+=['--mount','type=bind,src='+path+',dst='+path+',readonly']
command += [spec['image'],'/campaign/source/ascend_recording_server.py',*spec['arguments']]
result=subprocess.run(command,capture_output=True,text=True)
record={'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'checkpoint_sha256':spec['checkpoint_sha256'],'scope':'instrumented state capture, not performance measurement'}
(root/'launch.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
if result.returncode:raise RuntimeError('Recording service launch failed')
""".replace("PAYLOAD", repr(payload))
    result = run_python(remote_source, password, timeout=120)
    (ROOT / "recording-server-launch.json").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Peer recording service failed; inspect recording-server-launch.json")
    value = json.loads(result.stdout)
    print(json.dumps({"container": specification["name"], "returncode": value["returncode"], "port": 18000}))


if __name__ == "__main__":
    main()
