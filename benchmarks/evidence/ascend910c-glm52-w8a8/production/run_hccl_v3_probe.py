import json
import pathlib
import subprocess


def main():
    root = pathlib.Path('/tmp/aisim-910c-production-20261022/hccl-v3')
    root.mkdir(exist_ok=False)
    prefix = ['docker','exec','aisim-910c-production-20261022','timeout','300','python3','-m','torch.distributed.run','--nproc_per_node=2','--master_addr=127.0.0.1','--master_port=29928']
    negative = prefix + ['/campaign/hccl_graph_noop_probe.py','/campaign/hccl_v3.py','--output','/campaign/hccl-v3/noop.json','--ranks-per-node','2','--sizes','256','--operations','all_reduce','--modes','graph','--serving-setup','/campaign/serving-setup-nodes.json']
    with (root/'noop.log').open('w') as stream:
        result = subprocess.run(negative,stdout=stream,stderr=subprocess.STDOUT)
    record = json.loads((root/'noop.json').read_text())
    passed = result.returncode != 0 and record['status'] == 'failed' and 'Full-payload correctness failure' in record['error'] and not record['cases']
    report = {'negative_control_detected':passed,'negative_returncode':result.returncode,'negative_command':negative}
    (root/'probe-status.json').write_text(json.dumps(report,indent=2)+'\n')
    if not passed:
        raise RuntimeError('Graph no-op was not detected as a numerical failure')
    positive = prefix + ['/campaign/hccl_v3.py','--output','/campaign/hccl-v3/positive.json','--ranks-per-node','2','--sizes','256,100663296','--dtypes','bfloat16,float16,float32,int32','--serving-setup','/campaign/serving-setup-nodes.json']
    with (root/'positive.log').open('w') as stream:
        result = subprocess.run(positive,stdout=stream,stderr=subprocess.STDOUT)
    report.update(positive_returncode=result.returncode,positive_command=positive)
    (root/'probe-status.json').write_text(json.dumps(report,indent=2)+'\n')
    result.check_returncode()


if __name__ == '__main__':
    main()
