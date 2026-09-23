"""Run on Windows: refresh remote status and collect verified depth outputs."""
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = '/srv/encbank/comem_depth_sweep_20260915'
PYTHON = '/srv/encbank/Paper_Evolve/.venv/bin/python'
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-node1',
                         PYTHON, '-B', f'{REMOTE}/remote_status.py', '--collect'],
                        text=True, encoding='utf-8', capture_output=True, timeout=90, check=True)
report = json.loads(result.stdout)
(ROOT / 'STATUS.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
archive = ROOT / 'sweep_results.tar.gz'
subprocess.run(['scp', '-q', f"gpu-node1:{report['archive']}", str(archive)], timeout=90, check=True)
destination = ROOT / 'collected'
destination.mkdir(exist_ok=True)
with tarfile.open(archive, 'r:gz') as tar:
    tar.extractall(destination, filter='data')
for name, entries in report['models'].items():
    for j, item in entries.items():
        progress = item.get('progress.json', {})
        print(name, f'j={j}', 'VERIFIED' if item['verified'] else json.dumps(progress),
              'FAILED' if 'failure.json' in item else '')
print('Verified depths:', report['verified_depths'], '/', report['planned_depths'])
print('Queue:', report['queue'])
