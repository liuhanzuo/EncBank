"""Resume semantic decisions with a selected provider model and fixed prompt/seed.

Configure OPENAI_BASE_URL and OPENAI_API_KEY in this process's environment.
No credentials are written to records. API failures remain pending and retryable.
"""
import argparse, gzip, json, os, time
from common import ROOT, MODELS, ARMS, SHARDS, dump
from locomo_judge_reference import PROMPT, _endpoint, _request

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-index',type=int,required=True)
    ap.add_argument('--shard',type=int,required=True)
    ap.add_argument('--judge-model',required=True)
    a=ap.parse_args();cfg=MODELS[a.model_index]
    base=os.environ.get('OPENAI_BASE_URL');key=os.environ.get('OPENAI_API_KEY')
    assert base and key,'Set the previously authorized judge service in environment variables'
    out=ROOT/'results'/cfg['name']/f'shard{a.shard}'
    protocol=ROOT/'judge_protocol.json'
    assert protocol.exists(), 'Fix and calibrate a judge before scoring any experimental outputs'
    fixed=json.loads(protocol.read_text())
    assert fixed['transport']=='chat/completions', 'Use judge_gpt6.py for the selected official-CLI transport'
    assert fixed['model']==a.judge_model and fixed['calibration_passed'] and fixed['base_url'].rstrip('/')==base.rstrip('/')
    assert json.loads((out/'verified_summary.json').read_text())['verified']
    samples={}
    with gzip.open(ROOT/'samples'/cfg['name']/f'shard{a.shard}.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['benchmark']=='locomo':samples[r['id']]={k:r[k] for k in ('question','answers','extra')}
    dest=out/'judge_decisions.jsonl';seen={};failures=0
    if dest.exists():
        for line in dest.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);assert (r['id'],r['arm']) not in seen
            assert r['model']==a.judge_model and r['seed']==1, 'Cannot mix judge models in one run'
            seen[r['id'],r['arm']]=r
    with (out/'predictions.jsonl').open(encoding='utf-8') as src, dest.open('a',encoding='utf-8') as dst:
        for line in src:
            r=json.loads(line)
            if r['benchmark']!='locomo' or (r['id'],r['arm']) in seen:continue
            if r['status']!='ok': continue
            sample=samples[r['id']]
            if sample['extra']['is_abstention']:
                vote,raw=int(r['judge_score']),'LOCAL_ABSTENTION_RULE'
            else:
                prompt=PROMPT.format(question=sample['question'],gold=' OR '.join(map(str,sample['answers'])),pred=r['text'])
                vote,raw=_request(_endpoint(base),key,a.judge_model,prompt,1,4)
                if raw=='[API_FAILURE]':
                    failures+=1
                    with (out/'judge_failures.jsonl').open('a') as f:
                        f.write(json.dumps(dict(id=r['id'],arm=r['arm'],status='API_FAILURE',time=time.time()))+'\n')
                    # Stop quickly on a service outage, preserving earlier decisions.
                    if failures>=3:break
                    continue
            decision=dict(id=r['id'],arm=r['arm'],category=sample['extra']['category'],
                judge_correct=vote,judge_raw=raw,model=a.judge_model,seed=1)
            dst.write(json.dumps(decision,ensure_ascii=False)+'\n');dst.flush();seen[r['id'],r['arm']]=decision
            dump(out/'judge_progress.json',dict(decisions=len(seen),expected=len(samples)*len(ARMS),failures=failures))
    complete=len(seen)==len(samples)*len(ARMS)
    dump(out/'judge_status.json',dict(complete=complete,decisions=len(seen),expected=len(samples)*len(ARMS),failures=failures))
    print(json.dumps(dict(complete=complete,decisions=len(seen),failures=failures)))

if __name__=='__main__':main()
