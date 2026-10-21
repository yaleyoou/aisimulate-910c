import pathlib,subprocess,json
root=pathlib.Path("/mnt/raid/aisimulate-youxingye-20261022/recording")
results=[]
for batch in [1,10]:
 name="8192_1024_late_b"+str(batch)
 command=["python3",str(root/"source/ascend_state_sweep.py"),"--capture-dir",str(root/"capture"),"--case",name,"--concurrency",str(batch),"--output-tokens","1024","--arm-after-tokens","900"]
 with (root/(name+".log")).open("w") as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
 results.append({"case":name,"returncode":result.returncode});(root/"late-sweep-progress.json").write_text(json.dumps(results,indent=2))
 if result.returncode:raise RuntimeError("Late decode state capture failed")
