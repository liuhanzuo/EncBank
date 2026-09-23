import json
import config
import torch
from bm25_index import BM25Index
from comem.selectors import bm25_scores,iter_bm25_indices
def main():
    docs=json.loads((config.ROOT/'data/serving_32768.json').read_text())
    checked=0
    for d in docs:
        chunks=list(torch.tensor(d['source']).split(512));index=BM25Index(chunks)
        for q in d['queries']:
            assert index.scores(q['query'][:32])==bm25_scores([x.tolist() for x in chunks],q['query'][:32])
            assert index.select(q['query'][:32])==q['selected']==iter_bm25_indices(chunks,q['query'][:32],12,iter_hop_topk=2,iter_rounds=0)
            assert len(q['selected'])==12 and len(q['query'])==512
            checked+=1
    assert len(docs)==3 and checked==96
    (config.ROOT/'data/serving_verified.json').write_text(json.dumps(dict(complete=True,documents=3,request_templates=96,prebuilt_BM25_exact=True)))
    print('96 requests: prebuilt BM25 scores and multi-hop selections exactly match original.')
if __name__=='__main__':main()
