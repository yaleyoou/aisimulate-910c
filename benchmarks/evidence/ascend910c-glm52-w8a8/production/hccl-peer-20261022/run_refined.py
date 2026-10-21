import pathlib,subprocess,json
root=pathlib.Path("/tmp/aisim-910c-production-20261022")
output=root/"refined"
output.mkdir(exist_ok=False)
results=[]
for ranks in [2,4,8,16]:
 command=["docker","exec","aisim-910c-production-20261022","timeout","2400","python3","-m","torch.distributed.run","--nproc_per_node="+str(ranks),"--master_addr=127.0.0.1","--master_port=29922","/campaign/ascend_hccl.py","--output","/campaign/refined/hccl-"+str(ranks)+".json","--ranks-per-node",str(ranks),"--iterations","64","--warmup","20"]
 with (output/("hccl-"+str(ranks)+".log")).open("w") as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 results.append({"ranks":ranks,"returncode":result.returncode,"command":command})
 (output/"campaign-progress.json").write_text(json.dumps(results,indent=2))
 if result.returncode:raise SystemExit(result.returncode)
 data=json.loads((output/("hccl-"+str(ranks)+".json")).read_text())
 if sum(not row["repeatable_at_10_percent"] for row in data["cases"])*3>=len(data["cases"]):raise RuntimeError("Systematic timing variation")
