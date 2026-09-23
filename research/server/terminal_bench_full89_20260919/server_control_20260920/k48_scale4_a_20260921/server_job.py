"""One Slurm allocation supervises both the GPU holder and the Linux task controller."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from server_transport import save

H = Path(__file__).resolve().parent
P = json.loads((H / 'plan.json').read_text())


def main():
    assert os.environ.get('SLURM_JOB_ID') and sys.platform == 'linux'
    for key in ['rpc_root', 'results_root', 'controller_tmp', 'controller_cache', 'container_cache', 'sif_cache', 'ipc_root']:
        path = Path(P[key]).resolve()
        path.relative_to(Path('/srv/encbank').resolve())
        path.mkdir(parents=True, exist_ok=True)
    subprocess.run([P['harbor_python'], str(H / 'server_preflight.py')], cwd=H, check=True)
    holder = subprocess.Popen([sys.executable, '-u', str(H / 'holder.py')], cwd=H)
    with (H / 'owner.stdout.log').open('wb') as out, (H / 'owner.stderr.log').open('wb') as err:
        owner = subprocess.Popen([P['harbor_python'], '-u', str(H / 'server_owner.py')], cwd=H,
                                 env=dict(os.environ, PYTHONPATH=str(H)), stdout=out, stderr=err)
        def terminate(signum, _frame):
            save(H / 'allocation_signal.json', {'signal': signum, 'epoch': time.time()})
            if owner.poll() is None:
                owner.send_signal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
        while owner.poll() is None and holder.poll() is None:
            time.sleep(1)
        if holder.poll() is not None and owner.poll() is None:
            owner.send_signal(signal.SIGTERM)
        owner_code = owner.wait()
        if holder.poll() is None:
            # Normal controller shutdown requests worker stop after all task children close.
            try:
                holder_code = holder.wait(timeout=180)
            except subprocess.TimeoutExpired:
                holder.terminate()
                holder_code = holder.wait(timeout=60)
        else:
            holder_code = holder.wait()
    save(H / 'server_job_receipt.json', {'epoch': time.time(), 'owner_exit': owner_code,
         'holder_exit': holder_code, 'actual_parent_waits': True, 'server_only': True,
         'job_id': os.environ['SLURM_JOB_ID']})
    return 0 if owner_code == holder_code == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
