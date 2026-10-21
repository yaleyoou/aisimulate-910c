import json,os,pathlib,runpy,sys
rank=int(os.environ["LOCAL_RANK"])
world=int(os.environ["LOCAL_WORLD_SIZE"])
allowed=sorted(os.sched_getaffinity(0))
width=len(allowed)//world
if width<8:raise RuntimeError("Need eight disjoint CPUs per rank")
start=rank*width+(8 if width>=16 else 0)
selected=allowed[start:start+8]
os.sched_setaffinity(0,selected)
output=pathlib.Path(sys.argv[sys.argv.index("--output")+1])
output.with_name(output.stem+"-affinity-rank"+str(rank)+".json").write_text(json.dumps({"cpus":selected,"numa_locality":"not_claimed"}))
runpy.run_path("/campaign/ascend_hccl_gc.py",run_name="__main__")
