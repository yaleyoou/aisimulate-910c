import json
import os
import pathlib
import runpy
import sys


rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
allowed = sorted(os.sched_getaffinity(0))
partition = len(allowed) // world
if partition < 8:
    raise RuntimeError("Affinity protocol requires at least eight logical CPUs per rank")
offset = rank * partition + (8 if partition >= 16 else 0)
selected = allowed[offset:offset + 8]
os.sched_setaffinity(0, selected)
output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
output.with_name(output.stem + f"-affinity-rank{rank}.json").write_text(json.dumps({
    "rank": rank,
    "cpus": selected,
    "policy": "disjoint eight-logical-CPU sets; no NPU NUMA-locality claim",
}, indent=2) + "\n")
runpy.run_path(str(pathlib.Path(__file__).with_name("collect.py")), run_name="__main__")
