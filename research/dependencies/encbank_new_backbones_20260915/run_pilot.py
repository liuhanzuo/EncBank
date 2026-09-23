"""Small cross-backbone quality pilot, not a replacement for Table 1.

Runs the native-continuation/cache checks, 200 PG-19-only distillation steps,
and four paired quality arms. No selection of depth, checkpoints, or data is
based on evaluation scores. All raw generations and sample configurations save.
"""
import argparse, collections, gzip, json, os, platform, random, time, traceback, zlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from hybrid_reader import HybridReader
from encbank.selectors import iter_bm25_indices
from eval import ruler, longeval, longbench

HERE=Path(__file__).resolve().parent

def dump(path,obj):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)

def make_samples(tok, sink, n):
    rows=[]
    ruler._ESSAY_PATH='/srv/encbank/encbank_frozen_j12_20260912/data/pg19_essay.txt'
    assert ruler._load_essay_words()
    for length in ('8k','16k','32k'):
        for task in ('niah_single_2','niah_multikey_1','longeval'):
            for i in range(n):
                seed=20260915+zlib.crc32(f'{task}:{length}:{i}'.encode())
                rng=random.Random(seed)
                if task=='longeval':
                    prompt,answer,label,_=longeval.build_lines_prompt(longeval._LENGTH_TOKENS[length],tok,rng)
                    answers=[answer]; question=f'line {label}'; budget=16
                else:
                    prompt,answers,_=ruler._build_sample(task,ruler._LENGTH_TOKENS[length],tok,rng,None)
                    question=ruler._bare_question(prompt); budget=48
                rows.append(dict(id=f'{task}_{length}_{i:03}',task=task,length=length,prompt=prompt,
                                 question=question,answers=answers,budget=budget,seed=seed))
    qasper=Path('/srv/encbank/encbank_frozen_j12_20260912/data/longbench/qasper.jsonl')
    records=[json.loads(s) for s in qasper.read_text().splitlines() if s.strip()]
    # Fixed evenly spaced IDs cover the file; selection is independent of scores.
    for i in np.linspace(0,len(records)-1,3*n,dtype=int):
        row=records[i]
        rows.append(dict(id=f'qasper_{i:03}',task='qasper',length='native',prompt=longbench.format_prompt(row,'qasper'),
                         question=row['input'],answers=row['answers'],budget=128,dataset_index=int(i)))
    for row in rows:
        ids=tok.apply_chat_template([{'role':'user','content':row.pop('prompt')}],tokenize=True,
                                   add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if hasattr(ids, 'keys'):
            ids=ids['input_ids']
        assert isinstance(ids,list) and ids and all(isinstance(x,int) for x in ids)
        chunks=[ids[k:k+512] for k in range(0,len(ids),512)]
        source,query=chunks[:-1],chunks[-1]
        question_ids=tok.encode(row['question'],add_special_tokens=False)
        selected=iter_bm25_indices([torch.tensor(chunk,dtype=torch.long) for chunk in source],
                                  question_ids,12,iter_hop_topk=4,iter_rounds=0)
        selected=sorted(selected)
        assert source and query and selected
        assert all(0<=index<len(source) for index in selected)
        row.update(input_ids=ids,selected=selected,segments=[[sink]]+[source[k] for k in selected]+[query],
                   source_tokens=sum(map(len,source)),query_tokens=len(query))
    return rows

def train(reader,tok,sink,data,out,steps):
    params=reader.attach()
    ids=[]
    with open(data,encoding='utf-8') as f:
        for line in f:
            item=json.loads(line)
            ids.extend(tok.encode(item['text'],add_special_tokens=False))
            if len(ids)>=steps*2048:
                break
    assert len(ids)>=steps*2048,'Insufficient training tokens: never use evaluation inputs for distillation'
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.0)
    start=time.monotonic()
    with (out/'training.jsonl').open('w') as log:
        for step in range(steps):
            window=ids[step*2048:(step+1)*2048]
            segments=[[sink]]+[window[k:k+512] for k in range(0,2048,512)]
            with reader.adapter(False),torch.no_grad():
                teacher=reader.logits(reader.full_hidden([sink]+window),last=512)
                values,indices=teacher.topk(64,dim=-1)
                del teacher
                logp=F.log_softmax(values.float(),dim=-1)
                p=logp.exp()
            hidden=reader.cache_hidden(segments,grad=True)
            student=reader.logits(hidden,last=512)
            logq=F.log_softmax(student.gather(-1,indices).float(),dim=-1)
            q=logq.exp()
            loss=(.6*(p*(logp-logq)).sum(-1)+.4*(q*(logq-logp)).sum(-1)).mean()
            assert torch.isfinite(loss), 'Nonfinite distillation loss'
            loss.backward()
            gradnorm=torch.nn.utils.clip_grad_norm_(params,1.0)
            assert torch.isfinite(gradnorm)
            lr=1e-4*min((step+1)/20,1.0)*max((steps-step)/max(steps-20,1),0.0)
            for group in opt.param_groups: group['lr']=lr
            opt.step();opt.zero_grad(set_to_none=True)
            record={'step':step+1,'loss':float(loss.detach()),'grad_norm':float(gradnorm),'lr':lr,'elapsed_s':time.monotonic()-start}
            log.write(json.dumps(record)+'\n');log.flush()
            dump(out/'progress.json',{'phase':'training',**record,'total_steps':steps})
            if (step+1)%10==0: print(json.dumps(record),flush=True)
            del student,hidden,loss,logq,q,values,indices,logp,p
            if (step+1)%50==0: reader.save(out/f'adapter-step{step+1}.pt')
    reader.save(out/'adapter-final.pt')
    del opt,params
    torch.cuda.empty_cache()

def score(row,text):
    if row['task']=='longeval':return float(longeval.extract_prediction(text)==row['answers'][0])
    if row['task']=='qasper':return longbench.compute_f1_multi(text,row['answers'])
    return ruler._string_match_all_one(text,row['answers'])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--n',type=int,default=10)
    a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    assert os.environ.get('PYTHONHASHSEED')=='0'
    torch.set_num_threads(2);torch.manual_seed(20260915);random.seed(20260915);np.random.seed(20260915)
    out=HERE/'results'/Path(a.model).name/f"job_{os.environ['SLURM_JOB_ID']}"
    out.mkdir(parents=True,exist_ok=False)
    try:
        config=AutoConfig.from_pretrained(a.model,local_files_only=True)
        cls=AutoModelForCausalLM if config.model_type=='qwen3_5_text' else AutoModelForImageTextToText
        tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.model_max_length=10**9
        eos=tok.eos_token_id
        sink=tok.bos_token_id if tok.bos_token_id is not None else eos
        assert sink is not None
        textconfig=getattr(config,'text_config',config)
        j=round(.33*textconfig.num_hidden_layers)
        dump(out/'protocol.json',{'model':a.model,'revision_record':json.loads((Path(a.model)/'download_complete.json').read_text()),
             'j':j,'rule':'round(0.33 * L), fixed before outcomes','chunk_size':512,'topk':12,'seed':20260915,
             'chat_template':True,'enable_thinking':False,'sink':sink,'query':'last 512-token chunk of full native-chat prompt',
             'arms':['replay_base','cache_without_lora','replay_shared_lora','cache_lora'],
             'training':{'steps':a.steps,'window':2048,'data':'PG19 train 64 books; no benchmark supervision','rank':32,'alpha':32,
                         'targets':'all linear projections in suffix; prefix frozen','topk':64,'lambda':.6,'lr':1e-4},
             'n_per_synthetic_cell':a.n,'n_qasper':3*a.n,'pilot_only':True,'infrastructure_measurement':False,
             'gpu':torch.cuda.get_device_name(),'gpu_memory_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
             'host':platform.node(),'job':os.environ['SLURM_JOB_ID'],'torch':torch.__version__,'transformers':transformers.__version__})
        rows=make_samples(tok,sink,a.n)
        with gzip.open(out/'samples.jsonl.gz','wt',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        model=cls.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda').eval()
        reader=HybridReader(model,j)
        dump(out/'correctness.json',reader.validate(tok))
        train(reader,tok,sink,HERE/'data/pg19_train_64.jsonl',out,a.steps)
        # Verify decode after adapter training too, before recording any score.
        dump(out/'correctness_adapted.json',reader.validate(tok))
        eos_ids=set(getattr(model.generation_config,'eos_token_id',[]) or []) if isinstance(getattr(model.generation_config,'eos_token_id',None),list) else {eos}
        counts=collections.defaultdict(list)
        arms=[('replay_base','replay',False),('cache_without_lora','cache',False),('replay_shared_lora','replay',True),('cache_lora','cache',True)]
        with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
            for ri,row in enumerate(rows):
                for arm,mode,enabled in arms:
                    with reader.adapter(enabled),torch.inference_mode():
                        generated=reader.generate(row['segments'],mode,row['budget'],eos_ids)
                    text=tok.decode(generated,skip_special_tokens=True)
                    value=score(row,text)
                    key=f"{row['task']}:{row['length']}:{arm}"
                    counts[key].append(value)
                    f.write(json.dumps({'id':row['id'],'arm':arm,'generated_ids':generated,'text':text,'score':value,
                                        'task':row['task'],'length':row['length'],'answers':row['answers']},ensure_ascii=False)+'\n');f.flush()
                dump(out/'progress.json',{'phase':'evaluation','completed_samples':ri+1,'total_samples':len(rows)})
                print(json.dumps({'sample':ri+1,'total':len(rows),'id':row['id']}),flush=True)
        result={k:{'n':len(v),'mean':100*sum(v)/len(v)} for k,v in counts.items()}
        assert len(result)==40 and all(v['n']==(3*a.n if k.startswith('qasper:') else a.n) for k,v in result.items())
        dump(out/'summary.json',{'complete':True,'pilot_only':True,'cells':result})
        dump(out/'progress.json',{'phase':'complete','completed_samples':len(rows)})
    except BaseException:
        dump(out/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()})
        raise

if __name__=='__main__': main()
