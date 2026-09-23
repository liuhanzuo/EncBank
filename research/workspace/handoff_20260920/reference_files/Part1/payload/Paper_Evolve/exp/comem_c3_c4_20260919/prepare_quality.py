"""Freeze only the two requested quality blocks; no outcome-dependent selection."""
import collections,gzip,hashlib,json
from pathlib import Path
import config
import torch
from comem.selectors import iter_bm25_indices
def main():
    dest=config.ROOT/'data/quality.jsonl.gz'
    assert not dest.exists(),'Already prepared; do not overwrite fixed sample selection'
    base=Path('F:/Paper_Evolve/exp/comem_frozen_j12_20260912/collected/results')
    rows=[];origins=[]
    for length in ['8k','16k']:
        current=[]
        for p in sorted(base.glob(f'*/ruler_niah_multikey_1_{length}.inputs.jsonl.gz')):
            with gzip.open(p,'rt',encoding='utf-8') as f:current.extend(json.loads(s) for s in f)
            origins.append(str(p))
        assert len(current)==100 and len({r['index'] for r in current})==100
        for r in sorted(current,key=lambda x:x['index']):
            ids=r['input_ids'];source=((len(ids)-1)//512)*512
            rows.append(dict(id=f"multikey_{length}_{r['index']:03d}",benchmark='ruler',task='niah_multikey_1',
                cell='multikey_'+length,length=length,index=r['index'],input_ids=ids,
                question_ids=r['question_ids'],answers=r['answers'],source_tokens=source,
                original_source_tokens=r['source_tokens'],budget=r['max_new_tokens'],seed=r['seed']))
    path=Path('F:/Paper_Evolve/exp/comem_bge_matched_20260918/data/evaluation.jsonl.gz')
    with gzip.open(path,'rt',encoding='utf-8') as f:qa=[json.loads(s) for s in f]
    assert len(qa)==200 and all(not r['question_in_bank'] for r in qa)
    rows.extend(qa);origins.append(str(path))
    changes=[]
    for r in rows:
        chunks=list(torch.tensor(r['input_ids'][:r['source_tokens']]).split(512))
        selected=iter_bm25_indices(chunks,r['question_ids'],12,iter_hop_topk=4,iter_rounds=0)
        if 'selected' in r and r['selected']!=selected:changes.append(r['id'])
        r['selected']=selected
        r['source_sha256']=hashlib.sha256(json.dumps(r['input_ids'][:r['source_tokens']]).encode()).hexdigest()
        r['timing_subset']=r['index']<5
    assert len(rows)==len({r['id'] for r in rows})==400
    with gzip.open(dest,'wt',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report=dict(complete=True,n=400,cells=dict(collections.Counter(r['cell'] for r in rows)),
        origins=origins,selection='All saved multikey 8k/16k 100 each, all corrected Qasper200; no score selection',
        bm25_k=12,hop=4,source_order=True,qasper_boundary_corrected=True,
        updated_selection_after_boundary_fix=changes,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        quality_predictions=2400,timing_ids=[r['id'] for r in rows if r['timing_subset']])
    (config.ROOT/'data/quality_spec.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))
if __name__=='__main__':main()
