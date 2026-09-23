"""Full paper task/length support; fixed examples shared by all compatible arms."""
import argparse, collections, gzip, json, os, time
from pathlib import Path
import torch
from common import ROOT, MODELS, CHUNK, SHARDS, tokenizer, dump
from encbank.selectors import iter_bm25_indices
from eval import ruler, locomo
import run_accuracy as original

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-index', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    args=ap.parse_args(); cfg=MODELS[args.model_index]
    out=ROOT/('sample_smoke' if args.smoke else 'samples')/cfg['name']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        print('Samples already complete.'); return
    assert not any(out.glob('shard*.jsonl.gz')), 'Existing sample attempt: inspect before retry'
    assert os.environ.get('PYTHONHASHSEED')=='0'
    tok=tokenizer(cfg); torch.set_num_threads(2)
    ruler._ESSAY_PATH='/srv/encbank/encbank_frozen_j12_20260912/data/pg19_essay.txt'
    assert ruler._load_essay_words(), 'No fallback random haystack'
    records=locomo.build_locomo_samples(ROOT/'data/locomo10.json')
    assert len(records)==1986
    outputs=[gzip.open(out/f'shard{s}.jsonl.gz','wt',encoding='utf-8',compresslevel=3) for s in range(SHARDS)]
    counts=collections.Counter(); sequence=0; started=time.monotonic()
    def write(benchmark,task,length,index,prompt,question,answers,budget,extra):
        nonlocal sequence
        ids=tok.apply_chat_template([{'role':'user','content':prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if hasattr(ids,'keys'): ids=ids['input_ids']
        assert ids and isinstance(ids,list) and all(isinstance(i,int) for i in ids)
        chunks=[ids[i:i+CHUNK] for i in range(0,len(ids),CHUNK)]
        source,query=chunks[:-1],chunks[-1]
        qids=tok.encode(question,add_special_tokens=False)
        selected=sorted(iter_bm25_indices([torch.tensor(x,dtype=torch.long) for x in source],
            qids,12,iter_hop_topk=4,iter_rounds=0)) if source else []
        assert query and len(selected)<=12
        # Continue the newer-model screen's one-token BOS/EOS sink consistently.
        sink=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        segments=[[sink]]+[source[i] for i in selected]+[query]
        uid=f'{benchmark}:{task}:{length}:{index}'
        row=dict(id=uid, benchmark=benchmark, task=task, length=length, index=index,
            sequence=sequence, input_ids=ids, selected=selected, segments=segments,
            question=question, answers=answers, budget=budget, extra=extra,
            source_tokens=sum(map(len,source)), query_tokens=len(query), sink=sink)
        outputs[sequence%SHARDS].write(json.dumps(row,ensure_ascii=False)+'\n')
        counts[f'{benchmark}:{task}:{length}']+=1; sequence+=1
        if sequence%25==0:
            progress=dict(samples=sequence,target=7236,last_id=uid,elapsed_s=time.monotonic()-started)
            dump(out/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    try:
        for benchmark,task,length in original.cells():
            for index,prompt,question,answers,budget,extra in original.samples(benchmark,task,length,tok,0,1):
                # 48 is the previous no-LoRA budget; use it for every LongEval arm
                # here so paired new-model comparisons have identical stopping limits.
                write(benchmark,task,length,index,prompt,question,answers,budget,extra)
                if args.smoke: break
        for i,row in enumerate(records):
            write('locomo','qa','native',i,row['prompt'],row['question'],row['answers'],48,
                  dict(source_id=row['id'],category=row['category'],is_abstention=row['is_abstention']))
            if args.smoke: break
    finally:
        for f in outputs: f.close()
    expected=48 if args.smoke else 7236
    assert sequence==expected, (sequence,expected)
    totals=collections.Counter()
    for key,n in counts.items(): totals[key.split(':')[0]]+=n
    if not args.smoke:
        assert dict(totals)==dict(longbench=1150,babilong=2100,longeval=500,ruler=1500,locomo=1986)
    dump(out/'complete.json',dict(complete=True,model=cfg,samples=sequence,cells=dict(counts),
        totals=dict(totals),shards=SHARDS,native_chat=True,enable_thinking=False,
        generation_limits='RULER 48/VT60; LongEval all arms48; LongBench32/64/128; BABILong20; LoCoMo48',
        cohort='same task/length support as Table1; new-model tokenization; all arms share exact inputs',
        selection_caveat='Some natural QA IDs overlap the exploratory depth screen; not a held-out optimum claim'))
    print(json.dumps(dict(complete=True,samples=sequence,totals=dict(totals))),flush=True)

if __name__=='__main__': main()
