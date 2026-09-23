"""Score normalized saved LoCoMo answers with the single fixed GPT-6 protocol."""
import argparse, hashlib, json
from pathlib import Path
from codex_judge_client import ROOT, MODEL, call
from locomo_judge_reference import PROMPT, _REFUSAL_RE
from evaluation_scope import read_scope

def main():
    assert json.loads((ROOT/'locomo_scope.json').read_text())['judge_model']==MODEL
    ap=argparse.ArgumentParser()
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    protocol=json.loads((ROOT/'judge_protocol.json').read_text())
    assert protocol['model']==MODEL and protocol['status']=='READY' and protocol['calibration_passed'], 'GPT-6 connectivity/calibration must pass first'
    args.output.mkdir(parents=True,exist_ok=True)
    dest=args.output/'judge_decisions.jsonl';seen={}
    if dest.exists():
        for line in dest.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);k=(r['cohort'],r['arm'],r['id'])
            assert r['protocol_id']==protocol['protocol_id'] and r['model']==MODEL and k not in seen
            seen[k]=r
    expected=set();failure=None
    with args.inputs.open(encoding='utf-8') as src, dest.open('a',encoding='utf-8') as dst:
        for line in src:
            r=json.loads(line);k=(r['cohort'],r['arm'],r['id']);assert k not in expected;expected.add(k)
            stimulus={'question':r['question'],'gold':' OR '.join(map(str,r['answers'])),'pred':r['pred']}
            identity=hashlib.sha256(json.dumps(stimulus,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if k in seen:
                assert seen[k]['stimulus_digest']==identity;continue
            assert r.get('status','ok')=='ok' and isinstance(r['pred'],str)
            category=int(r['category'])
            if category==5:
                label='CORRECT' if not r['pred'].strip() or _REFUSAL_RE.search(r['pred']) else 'WRONG'
                raw='LOCAL_ABSTENTION_RULE';usage=None
            else:
                assert category in (1,2,3,4)
                call_id=hashlib.sha256(json.dumps([k,identity],ensure_ascii=False).encode()).hexdigest()
                result=call(PROMPT.format(**stimulus),args.output/'calls'/call_id)
                label=result['answer'].strip();raw=label;usage=result['usage']
                if not result['ok'] or label not in ('CORRECT','WRONG'):
                    failure=dict(cohort=r['cohort'],arm=r['arm'],id=r['id'],status='JUDGE_FAILED_UNSCORED',client_result=result)
                    break
            row=dict(cohort=r['cohort'],arm=r['arm'],id=r['id'],category=category,
                judge_correct=int(label=='CORRECT'),judge_raw=raw,model=MODEL,reasoning_effort='low',seed=None,
                protocol_id=protocol['protocol_id'],stimulus_digest=identity,usage=usage)
            dst.write(json.dumps(row,ensure_ascii=False)+'\n');dst.flush();seen[k]=row
            (args.output/'progress.json').write_text(json.dumps(dict(decisions=len(seen),last_id=r['id']))+'\n')
    status=dict(complete=failure is None and set(seen)==expected,decisions=len(seen),failure=failure,protocol_id=protocol['protocol_id'])
    (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status))

if __name__=='__main__':main()
