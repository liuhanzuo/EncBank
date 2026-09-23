"""Final CPU-only check of stored IDs, texts, references, scores, counts, and receipts."""
import collections,gzip,json,statistics,subprocess,sys
import config

def main():
    root=config.ROOT
    # Refresh only presentation of the report; preserve the original scientific files.
    r=subprocess.run([sys.executable,'-X','utf8','-B',str(root/'report.py')],capture_output=True,text=True)
    (root/'delivery_report_parent_exit.json').write_text(json.dumps(dict(actual_wait=True,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2),encoding='utf-8')
    assert r.returncode==0,r.stderr
    import common as original
    import torch
    from transformers import AutoTokenizer
    assert not torch.cuda.is_initialized()
    tok=AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    with gzip.open(root/'data/evaluation.jsonl.gz','rt',encoding='utf-8') as f:samples={r['id']:r for r in map(json.loads,f)}
    ranks={r['id']:r['ranking'] for r in json.loads((root/'data/rankings.json').read_text())}
    expected={('raw',6),('raw',8),('encbank',12)}
    data=[json.loads(l) for l in (root/'quality/process_00/records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(data)==1200
    keys=[(r['id'],r['method'],r['k_max'],r['rep']) for r in data];assert len(set(keys))==1200
    counts=collections.Counter((r['method'],r['k_max'],r['rep']) for r in data)
    assert counts==collections.Counter({(m,k,rep):200 for m,k in expected for rep in [-1,0]})
    formal=[r for r in data if not r['warmup']]
    for row in data:
        sample=samples[row['id']]
        assert row['status']=='ok' and row['context_sha256']==sample['context_sha256'] and row['source_sha256']==sample['source_sha256']
        assert row['selected']==sorted(ranks[row['id']][:row['k_max']])
        assert row['pack_tokens']==1+512*len(row['selected'])+len(sample['input_ids'])-sample['source_tokens']
        assert row['generated_tokens']==len(row['generated_ids'])
        assert tok.decode(row['generated_ids'],skip_special_tokens=True)==row['prediction']
        assert row['peak_reserved_bytes']<=27*2**30
        if not row['warmup']:
            assert abs(original.score(sample,row['prediction'])-row['score'])<1e-12
            assert row['generated_tokens']<=128 and row['rep']==0
        else:assert row['generated_tokens']==1 and row['score'] is None
    summary=json.loads((root/'summary.json').read_text())
    for method,k in expected:
        rows=[r for r in formal if (r['method'],r['k_max'])==(method,k)]
        assert {r['id'] for r in rows}==set(samples)
        assert len({r['context_sha256'] for r in rows})==148
        target=summary['arms'][f'{method}_k{k}']
        assert abs(target['F1']-100*statistics.mean(r['score'] for r in rows))<1e-12
        assert target['TTFT_median_ms']==statistics.median(r['ttft_ms'] for r in rows)
    for rep in range(3):
        p=root/'calibration'/f'process_{rep:02d}'
        rc=json.loads((p/'parent_exit.json').read_text());assert rc['actual_wait'] and rc['returncode']==0
        rs=[json.loads(l) for l in (p/'records.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rs)==560 and sum(not r['warmup'] for r in rs)==420 and all(r['score'] is None for r in rs)
    for p in [root/'quality/process_00/parent_exit.json',root/'report_run/parent_exit.json']:
        rc=json.loads(p.read_text());assert rc['actual_wait'] and rc['returncode']==0
    assert json.loads((root/'reservation_release.json').read_text())['reservation'] is None
    assert not torch.cuda.is_initialized()
    value=dict(complete=True,new_gpu_work=False,formal_answers=600,warmups_excluded=600,questions_per_arm=200,
        context_clusters=148,all_token_decodes_match=True,all_reference_F1_recomputed=True,all_selected_indices_match=True,
        receipts_valid=True,calibration_formal_records=1260,calibration_has_no_scores=True,local_reservation_released=True,
        errors=0,empty_answers=sum(not r['prediction'].strip() for r in formal),length_cap_count=sum(r['length_cap_reached'] for r in formal))
    (root/'delivery_verification.json').write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value))
if __name__=='__main__':main()
