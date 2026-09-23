"""Read the small archived training records; retain model checkpoints on the server."""
import json, math, tarfile
from pathlib import PurePosixPath
import config
from train_support import atomic_json


def main():
    files={}
    with tarfile.open(config.ROOT/'delivery/training_evidence.tar.gz','r:gz') as t:
        for member in t:
            assert member.isfile() and not PurePosixPath(member.name).is_absolute()
            assert '..' not in PurePosixPath(member.name).parts and member.name not in files
            files[member.name]=t.extractfile(member).read()
    summaries=[];sources=set()
    for arm in config.ARMS:
        prefix=f"runs/{arm['name']}/training/"
        meta=json.loads(files[prefix+'metadata.json']);status=json.loads(files[prefix+'status.json'])
        logs=[json.loads(l) for l in files[prefix+'train.jsonl'].decode().splitlines()]
        assert status['complete'] and status['step']==arm['steps'] and len(logs)==arm['steps']
        assert [r['step'] for r in logs]==list(range(1,arm['steps']+1))
        assert all(math.isfinite(r[k]) for r in logs for k in ('loss','lr','grad_norm'))
        assert all(r['token_cursor']==(8+r['step'])*4096 for r in logs)
        recipe=meta['recipe']
        for key,value in dict(j=12,seed=42,chunk=512,n_ctx=7,topk=64,lam=.6,loss='published',warmup=50,skip_windows=8).items():
            assert recipe[key]==value,(arm['name'],key)
        assert recipe['steps']==arm['steps'] and recipe['lr']==arm['lr']
        sources.add(meta['data']['source_sha256'])
        assert meta['data']['tokens']==7231927 and meta['data']['documents']==64
        if arm['mode']=='full':
            check=json.loads(files[prefix+'gradient_check.json'])
            assert check['teacher_frozen'] and check['all_student_gradients_finite']
            assert all(x>0 for x in check['nonzero_gradients'].values())
        else:
            assert recipe['rank']==arm['rank'] and recipe['alpha']==arm['alpha']
        summaries.append(dict(name=arm['name'],steps=len(logs),last_loss=logs[-1]['loss'],
            training_tokens=len(logs)*4096,trainable_parameters=meta['trainable_parameters'],
            training_elapsed_s=logs[-1]['elapsed_s'],precision=meta['precision'],
            checkpoint_remote=config.REMOTE+f"/runs/{arm['name']}/training/final"))
    assert len(sources)==1
    result=dict(complete=True,arms=5,same_pg19_token_order=True,all_planned_steps_complete=True,
        nonfinite_losses_or_grad_norms=0,all_full_gradient_checks_passed=True,arms_summary=summaries,
        timing_note='Shared Slurm allocation elapsed time, not infrastructure throughput')
    atomic_json(config.ROOT/'delivery/training_verification.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
