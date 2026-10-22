import json
import pathlib
import subprocess


def main():
    root = pathlib.Path('/tmp/aisim-910c-production-20261022/hccl-v3/full')
    root.mkdir(exist_ok=False)
    progress = {'status':'running','cases':[]}

    def run(command, label):
        progress['active'] = label
        (root/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        with (root/(label+'.log')).open('w') as stream:
            result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        progress['cases'].append({'label':label,'returncode':result.returncode,'command':command})
        (root/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        result.check_returncode()

    prefix=['docker','exec','-e','HCCL_IF_BASE_PORT=28500','-e','HCCL_CONNECT_TIMEOUT=120','-e','HCCL_EXEC_TIMEOUT=300','aisim-910c-production-20261022','timeout','1800','python3','-m','torch.distributed.run']
    try:
        for ranks in (2,4,8,16):
            run(prefix+['--nproc_per_node='+str(ranks),'--master_addr=127.0.0.1','--master_port=29929','/campaign/hccl_v3.py','--output',f'/campaign/hccl-v3/full/hccl-{ranks}.json','--ranks-per-node',str(ranks),'--serving-setup','/campaign/serving-setup-nodes.json'],f'hccl-{ranks}')
            data=json.loads((root/f'hccl-{ranks}.json').read_text())
            if sum(not row['repeatable_at_10_percent'] for row in data['cases'])*3>=len(data['cases']):
                raise RuntimeError('Investigate systematic timing variation before continuing')
        progress['status']='joining32'
        run(prefix+['--nnodes=2','--node_rank=1','--nproc_per_node=16','--master_addr=10.120.72.32','--master_port=29925','/campaign/hccl_v3.py','--output','/campaign/hccl-v3/full/hccl-32-peer.json','--ranks-per-node','16','--serving-setup','/campaign/serving-setup-nodes.json'],'hccl-32-peer')
        progress['status']='completed'
    except Exception as error:
        progress['status']='failed'
        progress['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        (root/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')


if __name__=='__main__':
    main()
