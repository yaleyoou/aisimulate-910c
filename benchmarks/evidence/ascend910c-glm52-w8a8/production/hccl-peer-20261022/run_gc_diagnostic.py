import pathlib,subprocess
root=pathlib.Path("/tmp/aisim-910c-production-20261022")
for policy in ["enabled","disabled"]:
 command=["docker","exec","aisim-910c-production-20261022","timeout","1200","python3","-m","torch.distributed.run","--nproc_per_node=8","--master_addr=127.0.0.1","--master_port=29922","/campaign/ascend_hccl_gc.py","--output","/campaign/gc-"+policy+".json","--ranks-per-node","8","--iterations","64","--warmup","20","--operations","all_reduce,all_gather","--modes","eager","--sizes","49152,3145728,12582912","--repeats","27","--gc-policy",policy]
 with (root/("gc-"+policy+".log")).open("w") as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 if result.returncode:raise SystemExit(result.returncode)
