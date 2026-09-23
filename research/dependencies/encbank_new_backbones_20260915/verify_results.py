"""CPU-only independent decoding/scoring of complete pilot results."""
import argparse, collections, gzip, json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from eval import ruler, longeval, longbench

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
a=p.parse_args()
protocol=json.loads((a.run/'protocol.json').read_text())
original=json.loads((a.run/'summary.json').read_text())
assert original['complete'] and original['pilot_only']
tok=AutoTokenizer.from_pretrained(protocol['model'],local_files_only=True)
with gzip.open(a.run/'samples.jsonl.gz','rt',encoding='utf-8') as f:
    samples={r['id']:r for r in map(json.loads,f)}
records=[json.loads(s) for s in (a.run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(records)==4*len(samples)
seen=set(); cells=collections.defaultdict(dict)
for record in records:
    key=(record['id'],record['arm'])
    assert key not in seen and record['arm'] in protocol['arms']
    seen.add(key); sample=samples[record['id']]
    assert record['answers']==sample['answers'] and record['task']==sample['task'] and record['length']==sample['length']
    assert 0<len(record['generated_ids'])<=sample['budget']
    text=tok.decode(record['generated_ids'],skip_special_tokens=True)
    assert text==record['text']
    if record['task']=='qasper': value=longbench.compute_f1_multi(text,sample['answers'])
    elif record['task']=='longeval': value=float(longeval.extract_prediction(text)==sample['answers'][0])
    else: value=ruler._string_match_all_one(text,sample['answers'])
    assert abs(value-record['score'])<1e-10
    cells[f"{record['task']}:{record['length']}:{record['arm']}"][record['id']]=value
assert set(original['cells'])==set(cells)
for key,items in cells.items():
    assert len(items)==original['cells'][key]['n']
    assert abs(100*np.mean(list(items.values()))-original['cells'][key]['mean'])<1e-9
rng=np.random.default_rng(20260915)
paired={}
for cell in sorted({':'.join(k.split(':')[:2]) for k in cells}):
    for other in ['replay_shared_lora','cache_without_lora','replay_base']:
        values=cells[cell+':cache_lora']; reference=cells[cell+':'+other]
        assert values.keys()==reference.keys()
        delta=np.array([values[k]-reference[k] for k in sorted(values)])*100
        draws=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
        paired[cell+':cache_lora-minus-'+other]={'n':len(delta),'difference':float(delta.mean()),
                                              'paired_bootstrap_95':np.quantile(draws,[.025,.975]).tolist()}
result={'verified':True,'pilot_only':True,'samples':len(samples),'predictions':len(records),
        'cells':original['cells'],'paired_differences':paired}
(a.run/'verified_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'verified':True,'samples':len(samples),'predictions':len(records)}))
