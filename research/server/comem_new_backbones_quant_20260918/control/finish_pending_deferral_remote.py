"""Finish receipt handling after Slurm returned Start=None for the pending cancellation."""
import fcntl,json,subprocess
from pathlib import Path
ROOT=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
with (ROOT.parent/'qcomem_gpu_admission.lock').open('a+b') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    hist=ROOT/'maintenance_history/local-bge-start-20260919'
    assert not (hist/'complete.json').exists()
    queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'],universal_newlines=True,timeout=30)
    assert not any(l.startswith('105748|') for l in queue.splitlines())
    accounting=subprocess.check_output(['sacct','-j','105748','-n','-P','-o','JobIDRaw,State,ExitCode,Start'],universal_newlines=True,timeout=30)
    parent=[l.split('|') for l in accounting.splitlines() if l.split('|')[0]=='105748']
    assert len(parent)==1 and parent[0][1].startswith('CANCELLED') and parent[0][3]=='None',parent
    p=ROOT/'runs/large-final-m0-s2/submission.json'
    assert json.loads(p.read_text())['job']=='105748'
    assert not (p.parent/'parent_exit.json').exists()
    assert json.loads((ROOT/'local_gpu_reservation.json').read_text())['token']=='comem-bge-matched-20260919'
    p.rename(hist/'original_submission.json')
    result=dict(deferred_job='105748',task='large-final-m0-s2',start=None,accounting=accounting,queue_after=queue,
        receipt_archived=True,resubmission='Automatic after local reservation releases; no started GPU task interrupted')
    (hist/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
