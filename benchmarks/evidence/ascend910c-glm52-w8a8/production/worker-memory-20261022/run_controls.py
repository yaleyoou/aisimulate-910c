import json
import subprocess
from pathlib import Path


def main():
    root = Path('/mnt/raid/aisimulate-youxingye-20261022/memory')
    progress = {'status': 'running', 'cases': []}
    for concurrency in (1, 10):
        command = [
            'docker', 'exec', 'aisim-glm52-memory-20261022', 'python3',
            '/campaign/source/ascend_serving_control.py',
            '--requests', f'/campaign/requests-b{concurrency}.json',
            '--output', f'/campaign/controls-b{concurrency}',
            '--allow-flush-cache', '--warmup', '1', '--repeats', '3',
        ]
        progress['active_command'] = command
        (root / 'control-progress.json').write_text(json.dumps(progress, indent=2) + '\n')
        with (root / f'controls-b{concurrency}.log').open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        progress['cases'].append({'concurrency': concurrency, 'returncode': result.returncode})
        if result.returncode:
            progress['status'] = 'failed'
            (root / 'control-progress.json').write_text(json.dumps(progress, indent=2) + '\n')
            raise SystemExit(result.returncode)
    progress['status'] = 'completed'
    progress.pop('active_command')
    (root / 'control-progress.json').write_text(json.dumps(progress, indent=2) + '\n')


if __name__ == '__main__':
    main()
