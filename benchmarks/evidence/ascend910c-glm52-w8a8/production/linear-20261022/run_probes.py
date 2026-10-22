import concurrent.futures
import json
import pathlib
import subprocess


def main():
    root = pathlib.Path('/tmp/aisim-910c-production-20261022/linear')
    common = ['--plan','/campaign/linear/plan.json','--serving-setup','/campaign/serving-setup-nodes.json']
    jobs = [
        ('cache32', ['python3','/campaign/linear/ascend_linear_v2.py',*common,'--device','0','--weight-copies','32','--case-filter','"m": 8,']),
        ('large8192', ['python3','/campaign/linear/ascend_linear_v2.py',*common,'--device','2','--case-filter','"m": 8192,']),
        ('graph-noop', ['python3','/campaign/hccl_graph_noop_probe.py','/campaign/linear/ascend_linear_v2.py',*common,'--device','4','--case-filter','"execution_mode": "graph"']),
    ]

    def run(job):
        name, arguments = job
        command=['docker','exec','aisim-910c-production-20261022','timeout','1200',*arguments,'--output',f'/campaign/linear/{name}.json']
        with (root/(name+'.log')).open('w') as stream: result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        report=json.loads((root/(name+'.json')).read_text())
        passed=(result.returncode!=0 and report['status']=='failed' and 'numerical check failed' in report['error']) if name=='graph-noop' else result.returncode==0 and report['status']=='completed'
        return {'name':name,'command':command,'returncode':result.returncode,'expected_outcome':passed,'cases':len(report['cases'])}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(run,jobs))
    (root/'probe-results.json').write_text(json.dumps(results,indent=2)+'\n')
    if not all(row['expected_outcome'] for row in results):raise RuntimeError(results)


if __name__=='__main__':main()
