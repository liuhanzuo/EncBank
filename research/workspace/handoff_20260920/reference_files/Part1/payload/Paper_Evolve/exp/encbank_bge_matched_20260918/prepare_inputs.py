"""CPU-only extra Qasper calibration inputs, exact context isolation, same BGE."""
import gzip,hashlib,io,json,os,random,re,tarfile,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import config
import torch
from transformers import AutoTokenizer
from encbank.selectors import DenseBGERetriever
from eval import longbench

def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def norm(x):return ' '.join(re.findall(r'\w+',x.lower()))
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)

def split_before_question(tok,prompt,question):
    enc=tok(prompt,add_special_tokens=True,return_offsets_mapping=True)
    start=prompt.rfind('Question: '+question);assert start>=0
    first=next(i for i,(a,b) in enumerate(enc['offset_mapping']) if b>start)
    boundary=(first//512)*512
    assert boundary>0 and all(b<=start for a,b in enc['offset_mapping'][:boundary])
    return enc['input_ids'],boundary

def main():
    out=config.ROOT/'data';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    torch.set_num_threads(2);torch.set_num_interop_threads(4)
    tok=AutoTokenizer.from_pretrained(config.MODEL,local_files_only=True)
    tok.model_max_length=10**9
    full=[json.loads(l) for l in config.QASPER.read_text(encoding='utf-8').splitlines()]
    with gzip.open(config.SAMPLES,'rt',encoding='utf-8') as f:
        saved={r['id']:r for r in map(json.loads,f) if r['cell']=='qasper'}
    assert len(full)==len(saved)==200
    evalrows=[];old_leakage=[]
    hashes={sha(r['context']) for r in full};normhashes={sha(norm(r['context'])) for r in full}
    prefixes={' '.join(norm(r['context']).split()[:64]) for r in full}
    for i,r in enumerate(full):
        row=dict(saved[f'qasper_{i:03d}']);prompt=longbench.format_prompt(r,'qasper')
        assert tok.encode(prompt,add_special_tokens=True)==row['input_ids']
        assert row['answers']==r['answers']
        # Determine whether any actual question text has leaked into the offline source bank.
        encoding=tok(prompt,add_special_tokens=True,return_offsets_mapping=True)
        qstart=prompt.rfind('Question: '+r['input'])
        assert qstart>=0
        source_end=((len(row['input_ids'])-1)//512)*512
        leakage=[k for k,(a,b) in enumerate(encoding['offset_mapping']) if k<source_end and b>qstart]
        if leakage:old_leakage.append(dict(id=row['id'],tokens=len(leakage),old_boundary=source_end))
        original_key=sha(json.dumps([row['input_ids'][k:k+512] for k in range(0,source_end,512)]))
        same_ids,boundary=split_before_question(tok,prompt,r['input']);assert same_ids==row['input_ids']
        row.update(context_sha256=sha(r['context']),normalized_context_sha256=sha(norm(r['context'])),
            source_sha256=sha(json.dumps([row['input_ids'][k:k+512] for k in range(0,boundary,512)])),
            original_source_sha256=original_key,original_source_tokens=source_end,
            full_context_chars=len(r['context']),source_tokens=boundary,question=r['input'],question_in_bank=False)
        evalrows.append(row)
    dump(out/'offline_boundary_audit.json',dict(affected_questions=old_leakage,count=len(old_leakage),
        correction='Complete 512-token chunks before Question marker are reusable; remaining prompt suffix is online query. Full input token IDs and prompt unchanged.'))
    # Existing local validation/heldout files overlap these contexts; use official extra training papers instead.
    archive=out/'qasper-train-dev-v0.3.tgz'
    url='https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz'
    if not archive.exists():
        import requests
        with requests.get(url,stream=True,timeout=(15,60)) as response:
            response.raise_for_status()
            with archive.with_suffix('.tmp').open('wb') as f:
                for block in response.iter_content(1024*1024):f.write(block)
        archive.with_suffix('.tmp').replace(archive)
    with tarfile.open(archive,'r:gz') as tar:
        member=next(x for x in tar.getmembers() if x.name.endswith('qasper-train-v0.3.json'))
        corpus=json.load(tar.extractfile(member))
    candidates=[];excluded=0
    for ident,r in corpus.items():
        context='\n'.join((s['section_name'] or '')+'\n'+'\n'.join(s['paragraphs']) for s in r['full_text'])
        normal=norm(context);prefix=' '.join(normal.split()[:64])
        if sha(context) in hashes or sha(normal) in normhashes or prefix in prefixes:excluded+=1;continue
        if not r['qas']:continue
        q=r['qas'][0] # fixed first question; no use of its answers
        candidates.append(dict(paper_id=ident,context=context,input=q['question'],question_id=q['question_id'],
            context_sha256=sha(context),normalized_context_sha256=sha(normal),words=len(normal.split())))
    # Stratify by evaluation context length; nearest candidate to each prespecified quantile, ties by seeded shuffle.
    rng=random.Random(config.SEED);rng.shuffle(candidates)
    lengths=sorted(len(norm(r['context']).split()) for r in full)
    chosen=[]
    for i in range(20):
        target=lengths[min(199,int((i+.5)/20*200))]
        k=min(range(len(candidates)),key=lambda k:abs(candidates[k]['words']-target))
        chosen.append(candidates.pop(k))
    cal=[]
    for i,r in enumerate(chosen):
        prompt=longbench.format_prompt(r,'qasper');ids,boundary=split_before_question(tok,prompt,r['input'])
        cal.append(dict(id=f'cal_qasper_{i:02d}',benchmark='qasper',cell='qasper',paper_id=r['paper_id'],
            question_id=r['question_id'],question=r['input'],context_sha256=r['context_sha256'],normalized_context_sha256=r['normalized_context_sha256'],
            input_ids=ids,question_ids=tok.encode(r['input'].strip(),add_special_tokens=False),budget=1,
            source_tokens=boundary,source_sha256=sha(json.dumps([ids[k:k+512] for k in range(0,boundary,512)]))))
    assert len({r['context_sha256'] for r in cal})==20
    assert not ({r['context_sha256'] for r in cal}&hashes)
    assert not ({r['normalized_context_sha256'] for r in cal}&normhashes)
    for name,rows in [('calibration',cal),('evaluation',evalrows)]:
        with gzip.open(out/(name+'.jsonl.gz'),'wt',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        dump(out/(name+'_ID_manifest.json'),[dict(id=r['id'],context_sha256=r['context_sha256'],
            source_sha256=r['source_sha256'],input_sha256=sha(json.dumps(r['input_ids'])),
            question_sha256=sha(r['question']),source_tokens=r['source_tokens']) for r in rows])
    prep=config.OLD/'bge_preparation';bge_manifest=json.loads((prep/'complete.json').read_text())
    encoder=DenseBGERetriever(str(config.BGE),device='cpu',dtype=torch.float32,batch_size=8)
    oldrank={r['id']:r for r in map(json.loads,(prep/'rankings.jsonl').read_text().splitlines())}
    indexes=out/'indexes';indexes.mkdir(exist_ok=True);rankings=[]
    for r in cal+evalrows:
        ids=r['input_ids'];chunks=list(torch.tensor(ids[:r['source_tokens']]).split(512));key=r['source_sha256']
        assert key==sha(json.dumps([c.tolist() for c in chunks]))
        path=indexes/(key+'.pt');index_s=None;reused=False
        if not path.exists():
            old=prep/(r.get('original_source_sha256',key)+'.pt')
            if old.exists():
                vectors=torch.load(old,map_location='cpu',weights_only=True)[:len(chunks)].clone();reused=True
            else:
                t=time.perf_counter();vectors=encoder.encode([tok.decode(c,skip_special_tokens=True) for c in chunks]);index_s=time.perf_counter()-t
            torch.save(vectors,path)
        else:vectors=torch.load(path,map_location='cpu',weights_only=True)
        start=time.perf_counter();q=encoder.encode([tok.decode(r['question_ids'],skip_special_tokens=True)],is_query=True)
        scores=(vectors@q[0]).tolist();ranking=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
        if r['id'] in oldrank:
            saved=oldrank[r['id']]['similarities'][:len(chunks)]
            assert ranking==sorted(range(len(saved)),key=lambda i:(-saved[i],i))
        rankings.append(dict(id=r['id'],source_sha256=key,ranking=ranking,index_bytes=vectors.numel()*vectors.element_size(),
            index_build_seconds=index_s,reused_existing_index=reused,query_search_seconds=time.perf_counter()-start))
        dump(out/'progress.json',dict(rankings=len(rankings),target=220))
    dump(out/'rankings.json',rankings)
    metadata=dict(complete=True,calibration=20,evaluation=200,calibration_contexts=20,evaluation_contexts=len(hashes),
        context_disjoint=True,normalization_disjoint=True,calibration_source=url,corpus_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        calibration_papers_excluded_for_overlap=excluded,selection='one first question per training paper; closest word length to 20 evaluation-length quantiles; seed20260918 tie order; no answer/score use',
        bge_revision=bge_manifest['revision'],bge_sha256=bge_manifest['model_hashes']['model.safetensors'],
        adapter_sha256=hashlib.sha256((config.ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest(),
        model_revision=(config.MODEL/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0],
        full_prompt_and_input_ids_unchanged=True,original_last_chunk_boundary_corrected=True,
        affected_old_questions=len(old_leakage),calibration_answers_excluded=True,no_evaluation_question_text_in_source_bank=True)
    dump(out/'complete.json',metadata);print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
