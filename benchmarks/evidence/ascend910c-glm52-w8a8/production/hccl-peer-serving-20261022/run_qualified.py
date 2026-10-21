import pathlib,subprocess,json
root=pathlib.Path("/tmp/aisim-910c-production-20261022/qualified")
root.mkdir(exist_ok=False)
results=[]
for ranks in [2,4,8,16]:
 command=["docker","exec","aisim-910c-production-20261022","timeout","1800","python3","-m","torch.distributed.run","--nproc_per_node="+str(ranks),"--master_addr=127.0.0.1","--master_port=29922","/campaign/ascend_hccl_qualified.py","--serving-setup","/campaign/serving-setup.json","--output","/campaign/qualified/hccl-"+str(ranks)+".json","--ranks-per-node",str(ranks)]
 with (root/("hccl-"+str(ranks)+".log")).open("w") as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 results.append({"ranks":ranks,"returncode":result.returncode,"command":command});(root/"campaign-progress.json").write_text(json.dumps(results,indent=2))
 if result.returncode:raise SystemExit(result.returncode)
 value=json.loads((root/("hccl-"+str(ranks)+".json")).read_text())
 if sum(not row["repeatable_at_10_percent"] for row in value["cases"])*3>=len(value["cases"]):raise RuntimeError("Inspect systematic timing variation before continuing")
