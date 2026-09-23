"""Independently decode and rescore each finished shard; never score OOM as zero."""
import argparse, collections, gzip, json, math
from common import ROOT, MODELS, ARMS, SHARDS, tokenizer, dump
from eval import ruler, longeval, longbench, locomo
import prepare_babilong as babi
from evaluation_scope import read_scope, shard_count, results_directory

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-index',type=int,required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();cfg=MODELS[a.model_index];tok=tokenizer(cfg);scope=read_scope()
    out=ROOT/('evaluation_smoke' if a.smoke else results_directory())/cfg['name']/f'shard{a.shard}'
    prep=ROOT/('sample_smoke' if a.smoke else 'samples')/cfg['name']
    complete=json.loads((out/'complete.json').read_text());assert complete['generation_complete']
    assert complete.get('scope_id')==scope['scope_id']
    inputs={};support=collections.Counter()
    with gzip.open(prep/f'shard{a.shard}.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['benchmark'] not in scope['benchmarks']:continue
            assert r['id'] not in inputs
            assert r['selected']==sorted(set(r['selected'])) and len(r['selected'])<=12
            chunks=[r['input_ids'][i:i+512] for i in range(0,len(r['input_ids']),512)]
            assert r['segments']==[[r['sink']]]+[chunks[i] for i in r['selected']]+[chunks[-1]]
            inputs[r['id']]={k:r[k] for k in ('benchmark','task','length','answers','budget','extra')}
            support[f"{r['benchmark']}:{r['task']}:{r['length']}"]+=1
    cells=collections.defaultdict(list);seen=set();oom=0;judge_pending=0
    with (out/'predictions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);key=(r['id'],r['arm']);assert key not in seen;seen.add(key)
            assert r['arm'] in ARMS and r['id'] in inputs
            sample=inputs[r['id']]
            assert all(r[k]==v for k,v in sample.items())
            value=None
            if r['status']=='OOM':
                assert r['score'] is None and r['generated_ids'] is None;oom+=1
            else:
                assert r['status']=='ok'
                assert 0<len(r['generated_ids'])<=r['budget']
                text=tok.decode(r['generated_ids'],skip_special_tokens=True);assert text==r['text']
                b=r['benchmark']
                if b=='ruler':value=ruler._string_match_all_one(text,r['answers'])
                elif b=='longeval':value=float(longeval.extract_prediction(text)==r['answers'][0])
                elif b=='longbench':value=longbench.compute_f1_multi(text,r['answers'])
                elif b=='babilong':value=babi.score_prediction(text,r['extra'],r['task'])
                else:
                    sc=locomo.score_sample(dict(pred=text,answers=r['answers'],**r['extra']))
                    assert sc==r['lexical'] and r['score'] is None
                    if r['extra']['is_abstention']: assert r['judge_score']==sc['acc']
                    else: assert r['judge_score'] is None;judge_pending+=1
                if value is not None: assert math.isclose(value,r['score'],abs_tol=1e-10) and 0<=value<=1
            c=f"{r['benchmark']}:{r['task']}:{r['length']}:{r['arm']}"
            cells[c].append(value)
    assert seen=={(uid,arm) for uid in inputs for arm in ARMS}
    assert len(inputs)==complete['samples'] and len(seen)==complete['records']
    assert len(inputs)==shard_count(json.loads((prep/'complete.json').read_text()),a.shard,scope)
    summary={}
    for c,vs in cells.items():
        assert len(vs)==support[c.rsplit(':',1)[0]]
        valid=[v for v in vs if v is not None]
        summary[c]=dict(n=len(vs),scored=len(valid),score=100*sum(valid)/len(vs) if len(valid)==len(vs) else None)
    dump(out/'verified_summary.json',dict(verified=True,model=cfg,shard=a.shard,samples=len(inputs),
        records=len(seen),oom_records=oom,pending_semantic_judgments=judge_pending,cells=summary,
        scope_id=scope['scope_id'],benchmarks=scope['benchmarks'],deferred_benchmarks=scope['deferred'],
        note='All selected-scope generations independently decoded and rescored; pending semantic judgments are not quality scores.'))
    print(json.dumps(dict(verified=True,samples=len(inputs),records=len(seen),oom=oom,judge_pending=judge_pending)),flush=True)

if __name__=='__main__':main()
