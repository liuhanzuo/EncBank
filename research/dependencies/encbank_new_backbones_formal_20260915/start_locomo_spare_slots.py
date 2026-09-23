"""Use three idle experiment slots while the final four-benchmark shard finishes."""
import datetime,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(args,**kwargs):
    r=subprocess.run(args,text=True,capture_output=True,**kwargs)
    assert r.returncode==0,(args,r.stdout,r.stderr)
    return r.stdout.strip()

def write(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)

def main():
    if '--release' in sys.argv:
        active=run(['squeue','-r','-j','45014','-h','-o','%i'])
        assert not active,active
        run(['scontrol','update','JobId=59046','ArrayTaskThrottle=4'])
        write(ROOT/'locomo_full_slots_restored.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),throttle=4))
        print('Four LoCoMo slots restored');return
    receipt=ROOT/'locomo_spare_slots.json'
    assert not receipt.exists(),'Inspect saved transition; do not repeat blindly'
    accounting=run(['sacct','-X','-j','45014','-n','-P','--format=JobID,State,ExitCode'])
    states={p[0]:p[1:] for line in accounting.splitlines() if (p:=line.split('|')) and p[0].startswith('45014_')}
    for i in range(7):assert states[f'45014_{i}'][:2]==['COMPLETED','0:0'],states
    assert states['45014_7'][0] in ('RUNNING','COMPLETING','COMPLETED'),states
    queued=run(['squeue','-r','-j','59046','-h','-o','%T']).split()
    assert len(queued)==8 and set(queued)=={'PENDING'},queued
    for name in ('Qwen3.5-9B','Qwen3.8-27B'):
        assert json.loads((ROOT/'training'/name/'complete.json').read_text())['steps']==4000
        assert json.loads((ROOT/'samples'/name/'complete.json').read_text())['samples']==7236
    record=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='PREPARING',
        reason='Seven completed evaluation tasks, one remaining; run existing LoCoMo array in three spare slots.',
        maximum_total_gpus=4,accounting=accounting)
    write(receipt,record)
    run(['scontrol','update','JobId=45014','ArrayTaskThrottle=1'])
    run(['scontrol','update','JobId=59046','ArrayTaskThrottle=3'])
    info=run(['scontrol','show','job','59046','-o'])
    assert 'ArrayTaskThrottle=3' in info,info
    run(['scontrol','update','JobId=59046','Dependency='])
    record['phase']='THREE_LOCOMO_SLOTS_RELEASED';write(receipt,record)
    batch='''#!/bin/bash
#SBATCH --job-name=midcache-locomo-release-slot
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/srv/encbank/encbank_new_backbones_formal_20260915/locomo-release-%j.out
#SBATCH --error=/srv/encbank/encbank_new_backbones_formal_20260915/locomo-release-%j.err
set -euo pipefail
cd /srv/encbank/encbank_new_backbones_formal_20260915
/srv/encbank/Paper_Evolve/.venv/bin/python -B start_locomo_spare_slots.py --release
'''
    job=run(['sbatch','--parsable','--dependency=afterany:45014_7'],input=batch).split(';')[0]
    assert job.isdigit(),job
    record.update(phase='COMPLETE',release_job=job)
    write(receipt,record)
    path=ROOT/'launch.json';launch=json.loads(path.read_text())
    launch.update(status='EVALUATING_FOUR_BENCHMARKS_AND_LOCOMO',locomo_release_job=job,
        gpu_schedule='one remaining four-benchmark shard + three LoCoMo slots; LoCoMo restores four after the last shard ends')
    write(path,launch)
    print(json.dumps(record))

if __name__=='__main__':main()
