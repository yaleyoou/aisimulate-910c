import pathlib,subprocess,json
root=pathlib.Path("/mnt/raid/aisimulate-youxingye-20261022/recording")
results=[]
for batch in [4,8,10]:
 name="8192_decode_b"+str(batch)
 command=["python3",str(root/"source/ascend_state_sweep.py"),"--capture-dir",str(root/"capture"),"--case",name,"--concurrency",str(batch)]
 with (root/(name+".log")).open("w") as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
 results.append({"case":name,"returncode":result.returncode});(root/"state-sweep-progress.json").write_text(json.dumps(results,indent=2))
 if result.returncode:raise RuntimeError("State sweep failed; inspect logs")
 case=root/"capture"/name
 records=[json.loads(line) for path in case.glob('rank-*.jsonl') for line in path.read_text().splitlines()]
 if len(records)!=48 or any(record['forward_mode'] not in {'DECODE','IDLE'} for record in records):raise RuntimeError("Incomplete or mixed-stage capture")
