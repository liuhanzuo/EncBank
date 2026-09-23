"""Fetch the experiment only after all four Slurm workers close successfully."""
from pathlib import Path
import json,subprocess,tarfile
HERE=Path(__file__).resolve().parent
REMOTE='/srv/encbank/comem_frozen_j12_20260912'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-node1']
def remote(cmd):
    return subprocess.check_output(SSH+[cmd],text=True,encoding='utf-8')
account=remote('sacct -X -j 25633 --noheader --parsable2 --format=JobID,State,ExitCode,Elapsed,NodeList,Start,End')
rows=[line.strip().split('|') for line in account.splitlines() if line.strip()]
workers={row[0]:row for row in rows if row[0] in {f'25633_{i}' for i in range(4)}}
assert set(workers)=={f'25633_{i}' for i in range(4)},workers
assert all(row[1]=='COMPLETED' and row[2]=='0:0' for row in workers.values()),workers
(HERE/'slurm_accounting.txt').write_text(account,encoding='utf-8')
archive=HERE/'completed_evidence.tar'
assert not archive.exists(),'Completed collection already exists; inspect it before repeating'
remote('tar -C '+REMOTE+' -cf '+REMOTE+'/completed_evidence.tar results smoke slurm-25617_4294967294.out slurm-25617_4294967294.err slurm-25625_4294967294.out slurm-25625_4294967294.err slurm-25633_0.out slurm-25633_0.err slurm-25633_1.out slurm-25633_1.err slurm-25633_2.out slurm-25633_2.err slurm-25633_3.out slurm-25633_3.err run_accuracy.py run.slurm')
subprocess.run(['scp','-o','BatchMode=yes','gpu-node1:'+REMOTE+'/completed_evidence.tar',str(archive)],check=True)
collected=HERE/'collected'; collected.mkdir(exist_ok=False)
with tarfile.open(archive) as tar: tar.extractall(collected,filter='data')
# Preserve executed source beside the copied source; expose results to the local
# aggregator without modifying predictions or failed smoke evidence.
import shutil
shutil.copytree(collected/'results',HERE/'results')
(HERE/'collection.json').write_text(json.dumps({'complete':True,'slurm_workers':workers,'archive_bytes':archive.stat().st_size,'executed_source_equal':(collected/'run_accuracy.py').read_bytes()==(HERE/'run_accuracy.py').read_bytes()},indent=2),encoding='utf-8')
assert (collected/'run_accuracy.py').read_bytes()==(HERE/'run_accuracy.py').read_bytes()
print('Collected all four completed workers.')
