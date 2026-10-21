import pathlib,subprocess,json
root=pathlib.Path("/mnt/raid/aisimulate-youxingye-20261022/recording")
results=[]
for tokens in [4096,2048,1024]:
 name=str(tokens)+"_early"
 command=["python3",str(root/"source/ascend_state_requests.py"),"--capture-dir",str(root/"capture"),"--case",name,"--input-tokens",str(tokens),"--output-tokens","16"]
 with (root/(name+".log")).open("w") as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
 results.append({"case":name,"returncode":result.returncode});(root/"prefill-sweep-progress.json").write_text(json.dumps(results,indent=2))
 if result.returncode:raise RuntimeError("Capture failed; preserve failure and inspect before continuing")
