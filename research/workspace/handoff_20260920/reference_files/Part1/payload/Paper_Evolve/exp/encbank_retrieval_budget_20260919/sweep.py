"""Retrieval-budget systems pilot; not an agent task-success benchmark."""
import copy,datetime,gc,hashlib,json,os,platform,random,subprocess
from functools import partial
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
import common_benchmark as cb
from engine import Engine
from bm25_index import BM25Index
from encbank import Encbank
ROOT=Path(__file__).resolve().parent
KS=[12,24,32,48]
cb.ROOT=ROOT

def make_docs(base,k):
    docs=copy.deepcopy(base)
    for doc in docs:
        chunks=[doc['source'][i:i+512] for i in range(0,len(doc['source']),512)]
        index=BM25Index(chunks)
        for query in doc['queries']:
            selected=index.select(query['query'][:32],k,2)
            assert len(selected)==k and len(set(selected))==k
            if k==12:assert selected==query['selected']
            query['selected']=selected
    return docs

@torch.inference_mode()
def validate(model,tok,by_k):
    checks=[]
    for k,docs in by_k.items():
        for method in ['raw','h_gpu']:
            engine=Engine(model,tok,docs,method,1,k=k)
            data=engine.infer([(0,0)],count=2,capture=True)[2]
            ids=docs[0]['queries'][0]['selected']
            if method=='h_gpu':
                sink=engine.sink;states=[engine.bank[i:i+1] for i in ids]
            else:
                sink=engine.reader.write_chunk([tok.bos_token_id])
                states=[engine.reader.write_chunk(engine.chunks[0][i]) for i in ids]
            qh,bottom,qp=engine.reader.write_prefill(docs[0]['queries'][0]['query'])
            logits,top,length=engine.reader.read_prefill(sink,states,qh)
            assert length==k*512+513
            reference=[logits[:,-1].float().cpu()]
            token=int(logits[:,-1].argmax(-1).item())
            logits=Encbank.decode_step(engine.reader,token,bottom,top,qp,length)
            reference.append(logits[:,-1].float().cpu())
            actual=torch.stack(data['logits_trace']);expected=torch.stack(reference)
            delta=actual-expected
            check=dict(k=k,method=method,pack_positions=length,max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()),argmax_changes=int(actual.argmax(-1).ne(expected.argmax(-1)).sum()))
            checks.append(check);cb.dump(ROOT/'correctness_progress.json',checks)
            assert check['max_abs']<.5 and check['rms']<.1 and check['argmax_changes']==0,check
            del engine,sink,states,qh,bottom,top,logits,actual,expected,delta
            gc.collect();torch.cuda.empty_cache()
    cb.dump(ROOT/'correctness.json',dict(passed=True,checks=checks,scope='two decoding steps against original Encbank read_prefill/decode_step for every budget and method; not task quality'))

def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(42)
    prop=torch.cuda.get_device_properties(0)
    assert (prop.major,prop.minor)==(10,3) and prop.total_memory>200*2**30
    torch.cuda.set_per_process_memory_fraction(128*2**30/prop.total_memory,0)
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa=lambda *args,**kwargs:False
    uuid=str(prop.uuid);uuid=uuid if uuid.startswith('GPU-') else 'GPU-'+uuid
    cb.dump(ROOT/'environment.json',dict(name=prop.name,requested_entry='B300',compute_capability=[prop.major,prop.minor],gpu_uuid=uuid,total_memory_bytes=prop.total_memory,
        torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,node=platform.node(),job=os.environ['SLURM_JOB_ID'],cpu_threads=4,cpu_affinity=sorted(os.sched_getaffinity(0)),
        allocator_cap_bytes=128*2**30,model=cb.MODEL,adapter=cb.ADAPTER,adapter_sha256=hashlib.sha256((Path(cb.ADAPTER)/'adapter_model.safetensors').read_bytes()).hexdigest()))
    token=AutoTokenizer.from_pretrained(cb.MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(cb.MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    from peft import PeftModel
    from unittest.mock import patch
    with patch('peft.tuners.lora.model.dispatch_torchao',return_value=None):wrapper=PeftModel.from_pretrained(model,cb.ADAPTER,autocast_adapter_dtype=True)
    model=wrapper.base_model.model.eval();model.requires_grad_(False);token.bos_token_id=model.config.bos_token_id
    base=json.loads((ROOT/'workloads.json').read_text());by_k={k:make_docs(base,k) for k in KS}
    templates=cb.profiles(base)['diverse']
    cb.dump(ROOT/'access_patterns.json',dict(ks=KS,hop=2,source_tokens=32768,query_tokens=512,output_tokens=32,templates=templates,
        selected={str(k):[by_k[k][di]['queries'][qi]['selected'] for di,qi in templates] for k in KS}))
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]):
        cb.dump(ROOT/'status.json',dict(phase='CORRECTNESS'));validate(model,token,by_k)
        points=[(k,method,c) for k in KS for method in ['raw','h_gpu'] for c in [1,4,16]]
        random.Random(20260920).shuffle(points);cb.dump(ROOT/'order.json',points)
        for k,method,c in points:
            cb.Engine=partial(Engine,k=k,hop=2)
            cb.run_point(model,token,by_k[k],f'k{k}',templates,method,c,uuid,64)
    cb.dump(ROOT/'complete.json',dict(complete=True,points=24,requests_per_point=64,at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    cb.dump(ROOT/'status.json',dict(phase='COMPLETE',points=24))

if __name__=='__main__':
    try:main()
    except BaseException as exc:cb.dump(ROOT/'failure.json',dict(error=repr(exc)));raise
