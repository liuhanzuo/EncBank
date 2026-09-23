"""CPU-only fixture preparation; retain existing quality samples verbatim."""
import gzip,json
import config
from pathlib import Path

def main():
    from transformers import AutoTokenizer
    import torch
    from encbank.selectors import iter_bm25_indices
    from controlled_train_core import atomic_json
    out=config.ROOT/'data';out.mkdir(exist_ok=True)
    rows=[]
    with gzip.open(config.LONGEVAL,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);r['benchmark']='longeval';rows.append(r)
    assert len(rows)==600
    with gzip.open(config.QASPER,'rt',encoding='utf-8') as f:
        qa=[json.loads(line) for line in f if line.strip()]
    qa=[r for r in qa if r['benchmark']!='longeval'];assert len(qa)==200
    rows+=qa;assert len({r['id'] for r in rows})==800
    with gzip.open(out/'quality.jsonl.gz','wt',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    tok=AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    books=[]
    with open(config.DATA,encoding='utf-8') as f:
        for bi,line in enumerate(f):
            ids=tok.encode(json.loads(line)['text'],add_special_tokens=False)
            if len(ids)>=131072+512+31*64:books.append((bi,ids))
            if len(books)==3:break
    assert len(books)==3
    for length in config.SOURCE_LENGTHS:
        docs=[]
        for bi,ids in books:
            source=ids[:length];chunks=list(torch.tensor(source).split(512));queries=[]
            for qi in range(32):
                query=ids[length+qi*64:length+qi*64+512]
                ix=iter_bm25_indices(chunks,query[:32],12,iter_hop_topk=2,iter_rounds=0)
                assert len(query)==512 and len(ix)==12
                queries.append(dict(query=query,selected=ix))
            docs.append(dict(book_index=bi,source=source,queries=queries))
        atomic_json(out/f'serving_{length}.json',docs)
    atomic_json(out/'complete.json',dict(complete=True,quality_rows=800,longeval=600,qasper=200,
        serving_docs=3,queries_per_doc=32,lengths=config.SOURCE_LENGTHS))

if __name__=='__main__':main()
