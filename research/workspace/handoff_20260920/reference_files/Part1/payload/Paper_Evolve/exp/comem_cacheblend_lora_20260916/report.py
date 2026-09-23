"""Independently decode/rescore complete paired predictions and export raw data."""
import collections,csv,gzip,json,sys
import config
sys.path.insert(0,str(config.FOLLOW))
import common as original
from transformers import AutoTokenizer
ARMS=('cacheblend_frozen','cacheblend_own_lora','comem_frozen','comem_own_lora')
def main():
    root=config.ROOT;out=root/'evaluation'
    assert json.loads((out/'complete.json').read_text())['records']==2000
    training=json.loads((root/'training/complete.json').read_text())
    assert training['steps']==4000 and training['training_tokens']==16384000
    tok=AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    with gzip.open(config.SAMPLES,'rt',encoding='utf-8') as f:inputs={r['id']:r for r in map(json.loads,f)}
    assert len(inputs)==500
    seen={};values=collections.defaultdict(list)
    for line in (out/'predictions.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);key=(r['id'],r['arm'])
        assert key not in seen and r['id'] in inputs and r['arm'] in ARMS
        sample=inputs[r['id']]
        assert r['cell']==sample['cell'] and r['selected']==sample['selected'] and r['budget']==sample['budget']
        assert r['status']=='ok' and 0<len(r['generated_ids'])<=sample['budget']
        text=tok.decode(r['generated_ids'],skip_special_tokens=True);assert text==r['prediction']
        score=original.score(sample,text);assert abs(score-r['score'])<1e-10
        seen[key]=r;values[(r['arm'],r['cell'])].append(score)
    assert set(seen)=={(uid,a) for uid in inputs for a in ARMS}
    table={}
    for arm in ARMS:
        table[arm]={}
        for cell,n in [('longeval_8k',100),('longeval_16k',100),('longeval_32k',100),('qasper',200)]:
            vals=values[(arm,cell)];assert len(vals)==n
            table[arm][cell]=100*sum(vals)/len(vals)
        table[arm]['longeval_macro']=sum(table[arm][f'longeval_{n}k'] for n in (8,16,32))/3
    gains={cell:table['cacheblend_own_lora'][cell]-table['cacheblend_frozen'][cell] for cell in table['cacheblend_frozen']}
    metadata=json.loads((root/'training/metadata.json').read_text())
    summary=dict(complete=True,verified=True,samples=500,records=2000,table=table,
        cacheblend_training_gain_pp=gains,training=training,recipe=metadata,
        interpretation='Same data/parameter/token/step budget and exact paired evaluation; not identical training FLOPs or a native CacheBlend system reproduction')
    original.dump(root/'summary.json',summary)
    with (root/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['method','longeval_8k','longeval_16k','longeval_32k','longeval_macro','qasper']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for arm in ARMS:writer.writerow(dict(method=arm,**table[arm]))
    with (root/'per_sample_scores.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['id','cell',*ARMS]);writer.writeheader()
        for uid,row in inputs.items():writer.writerow(dict(id=uid,cell=row['cell'],**{a:seen[(uid,a)]['score'] for a in ARMS}))
    with (root/'training_curve.csv').open('w',newline='',encoding='utf-8-sig') as f:
        logs=[json.loads(l) for l in (root/'training/train.jsonl').read_text().splitlines()]
        writer=csv.DictWriter(f,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs)
    original.dump(root/'COMPLETE.json',dict(complete=True,verified_records=2000,training_steps=4000,
        files=['results.csv','per_sample_scores.csv','training_curve.csv','summary.json','evaluation/predictions.jsonl','training/final/adapter_model.safetensors']))
    print(json.dumps(dict(complete=True,table=table,cacheblend_training_gain_pp=gains)))
if __name__=='__main__':main()
