"""Fresh-allocation synthetic GDN check and microbenchmark, zero model requests."""
import ast
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def main():
    import torch
    from vllm_gdn_adapter import VllmGdnDecode
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    assert os.environ.get('ENCBANK_ISOLATED_KERNEL_PILOT')=='1'
    torch.set_num_threads(1);torch.manual_seed(4202)
    p=Path('/srv/encbank/Paper_Evolve/.venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py')
    # Execute only the two audited pure PyTorch reference definitions, using this
    # process's torch; never import a second installed transformers package.
    tree=ast.parse(p.read_text());wanted={'l2norm','torch_recurrent_gated_delta_rule'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes}==wanted
    for n in nodes:n.decorator_list=[]
    ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    reference=ns['torch_recurrent_gated_delta_rule'];adapter=VllmGdnDecode()
    def timed(fn,args,repeats=30):
        for _ in range(5):fn(*args)
        torch.cuda.synchronize();samples=[]
        for _ in range(3):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(repeats):fn(*args)
            b.record();b.synchronize();samples.append(a.elapsed_time(b)/repeats)
        return statistics.median(samples)
    cases=[];started=time.time()
    with torch.inference_mode():
        for dtype in [torch.float32,torch.bfloat16]:
            for batch,heads,kdim,vdim in [(1,2,32,16),(1,48,128,128),(8,48,128,128)]:
                q=torch.randn(batch,1,heads,kdim,device='cuda',dtype=dtype)
                k=torch.randn_like(q);v=torch.randn(batch,1,heads,vdim,device='cuda',dtype=dtype)
                g=-torch.rand(batch,1,heads,device='cuda');beta=torch.rand(batch,1,heads,device='cuda',dtype=dtype)
                h=torch.randn(batch,heads,kdim,vdim,device='cuda')*.05
                expected=h.clone();observed=h.clone();out_err=state_err=0.
                for step in range(16):
                    before=observed.clone();input_state=observed
                    yy,expected=reference(q,k,v,g,beta,expected,True,True)
                    y,observed=adapter(q,k,v,g,beta,observed,True,True)
                    torch.testing.assert_close(input_state,before,rtol=0,atol=0)
                    torch.testing.assert_close(y,yy,atol=2e-3 if dtype==torch.bfloat16 else 2e-4,rtol=2e-2 if dtype==torch.bfloat16 else 2e-4)
                    torch.testing.assert_close(observed,expected,atol=2e-4,rtol=2e-3)
                    out_err=max(out_err,float((y.float()-yy.float()).abs().max()))
                    state_err=max(state_err,float((observed-expected).abs().max()))
                args=(q,k,v,g,beta,h,True,True)
                baseline_ms=timed(reference,args);adapter_ms=timed(adapter,args)
                cases.append(dict(dtype=str(dtype),batch=batch,heads=heads,kdim=kdim,vdim=vdim,
                    recurrent_steps=16,max_output_error=out_err,max_state_error=state_err,
                    reference_ms=baseline_ms,adapter_ms=adapter_ms,speedup=baseline_ms/adapter_ms))
    result=dict(status='PASS',epoch=time.time(),seconds=time.time()-started,job_id=os.environ['SLURM_JOB_ID'],
        device=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,cases=cases,
        reference_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),gpu_kernel_executed=True,
        model_calls=0,benchmark_attempts=0,production_approved=False,
        limitation='Synthetic single-layer GDN only, includes state-layout conversion. Not full-model speed or quality validation.')
    out=Path(__file__).resolve().parent/'gpu_probe_result.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
