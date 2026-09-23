"""Bind already completed independently adapted KV comparison; no model execution."""
import gzip,hashlib,json,statistics
from pathlib import Path
import config
def main():
    root=Path('F:/Paper_Evolve/exp/encbank_cacheblend_lora_20260916')
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cb=load(root/'training/metadata.json');cm=load(root/'encbank_training_metadata.json')
    fields=['j','rank','alpha','chunk','n_ctx','steps','skip_windows','seed','topk','lam','lr','warmup']
    assert all(cb['recipe'][k]==cm['recipe'][k] for k in fields)
    assert cb['source']['source_sha256']==cm['recipe']['source_sha256']
    assert cb['source']['tokens']==cm['recipe']['data_tokens']==7231927
    complete=load(root/'COMPLETE.json');assert complete['complete'] and complete['verified_records']==2000
    with gzip.open(config.SAMPLES,'rt',encoding='utf-8') as f:samples={r['id']:r for r in map(json.loads,f)}
    rows=[json.loads(l) for l in (root/'evaluation/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==2000 and len({(r['id'],r['arm']) for r in rows})==2000
    for r in rows:
        assert r['status']=='ok' and r['selected']==samples[r['id']]['selected'] and r['budget']==samples[r['id']]['budget']
    groups={a:[r for r in rows if r['arm']==a] for a in ['cacheblend_own_lora','encbank_own_lora']}
    table={}
    for a,rs in groups.items():
        assert len(rs)==500
        table[a]=dict(LongEval=100*statistics.mean(r['score'] for r in rs if r['cell'].startswith('longeval')),
            Qasper=100*statistics.mean(r['score'] for r in rs if r['cell']=='qasper'))
    result=dict(complete=True,new_training=False,new_generation=False,matched_training_fields={k:cb['recipe'][k] for k in fields},
        data_sha256=cb['source']['source_sha256'],training_tokens=16384000,parameter_count=58195968,
        matched_source_ids=500,selected_chunk_order_and_budget_equal=True,table=table,
        sample_file=str(config.SAMPLES),sample_sha256=hashlib.sha256(config.SAMPLES.read_bytes()).hexdigest(),
        evaluation_protocol=load(root/'evaluation/protocol.json'),
        limits='Same data/steps/parameters/loss/optimizer budget, not equal FLOPs/GPU-hours. Single training seed; task tradeoffs. Existing runtime versions differ; do not infer paired system speedups from separate process timing.')
    out=config.ROOT/'existing_analysis/kv_alignment.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(table,indent=2))
if __name__=='__main__':main()
