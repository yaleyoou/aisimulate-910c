"""Capture only performance-related environment and observed worker CPU masks."""

import hashlib
import json
import pathlib
import subprocess


def main():
    source = """
import json,pathlib,re
allowed={"PYTORCH_NPU_ALLOC_CONF","SGLANG_SET_CPU_AFFINITY","STREAMS_PER_DEVICE",
"HCCL_BUFFSIZE","DEEPEP_HCCL_BUFFSIZE","TASK_QUEUE_ENABLE","SGLANG_NPU_USE_MULTI_STREAM",
"SGLANG_ENABLE_OVERLAP_PLAN_STREAM","SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
"HCCL_ALGO","HCCL_OP_EXPANSION_MODE","HCCL_DETERMINISTIC","HCCL_SOCKET_IFNAME","GLOO_SOCKET_IFNAME"}
environment=None
affinity={}
for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try:
        arguments=path.read_bytes().split(b'\\0')
        if b'sglang.launch_server' in arguments:
            entries=(path.parent/'environ').read_bytes().split(b'\\0')
            values={entry.partition(b'=')[0].decode():entry.partition(b'=')[2].decode()
                    for entry in entries if entry.partition(b'=')[0].decode() in allowed}
            environment={name:values.get(name) for name in sorted(allowed)}
        title=arguments[0].decode()
        if title.startswith('sglang::scheduler'):
            rank=re.search(r'_TP(\\d+)',title)
            status=(path.parent/'status').read_text()
            encoded=next(line.split(':',1)[1].strip() for line in status.splitlines()
                         if line.startswith('Cpus_allowed_list:'))
            cpus=[]
            for block in encoded.split(','):
                start,separator,end=block.partition('-')
                cpus.extend(range(int(start),int(end)+1) if separator else [int(start)])
            affinity[rank.group(1)]=cpus
    except (FileNotFoundError,PermissionError):
        pass
if environment is None or sorted(map(int,affinity))!=list(range(16)):
    raise RuntimeError('Expected live United launcher and all16 workers')
print(json.dumps({'schema':'sglang-runtime-setup/v1','environment':environment,
                  'environment_scope':'Launcher exports, verified against run-united.sh; worker proctitle can obscure /proc/environ.',
                  'worker_cpu_affinity':affinity,'scope':'Observed United16-workers/node; not an inferred NUMA policy.'}))
"""
    result = subprocess.check_output(["docker", "exec", "glm52-united-v0522", "python3", "-c", source], text=True)
    value = json.loads(result)
    launch = pathlib.Path("/home/youxingye/United deployment/service/run-united.sh")
    value["launcher_path"] = str(launch)
    value["launcher_sha256"] = hashlib.sha256(launch.read_bytes()).hexdigest()
    pathlib.Path(__file__).with_name("serving-setup.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"workers": len(value["worker_cpu_affinity"]), "environment": value["environment"]}))


if __name__ == "__main__":
    main()
