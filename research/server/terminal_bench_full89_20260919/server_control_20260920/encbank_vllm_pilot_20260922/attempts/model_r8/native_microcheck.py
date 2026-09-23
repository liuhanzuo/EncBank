"""Numerical and aliasing gates before loading the isolated full model."""
import statistics
import torch
from native_layout_adapter import NativeLayoutGdn

def check_native(reference):
    torch.manual_seed(4211)
    rows=[]
    def timed(fn, args):
        for _ in range(8): fn(*args)
        torch.cuda.synchronize()
        vals=[]
        for _ in range(3):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(30): fn(*args)
            b.record(); b.synchronize(); vals.append(a.elapsed_time(b)/30)
        return statistics.median(vals)
    for dtype in [torch.float32,torch.bfloat16]:
        for batch,heads,kdim,vdim in [(1,2,32,16),(1,48,128,128),(8,48,128,128)]:
            q=torch.randn(batch,1,heads,kdim,device='cuda',dtype=dtype)
            k=torch.randn_like(q);v=torch.randn(batch,1,heads,vdim,device='cuda',dtype=dtype)
            g=-torch.rand(batch,1,heads,device='cuda');beta=torch.rand(batch,1,heads,device='cuda',dtype=dtype)
            h=torch.randn(batch,heads,kdim,vdim,device='cuda')*.05
            for fuse in [False,True]:
                adapter=NativeLayoutGdn(fuse_hf_norm=fuse)
                expected=h.clone();observed=h.clone();oe=se=0.
                for _ in range(16):
                    before=observed.clone();input_state=observed
                    yy,expected=reference(q,k,v,g,beta,expected,True,True)
                    y,observed=adapter(q,k,v,g,beta,observed,True,True)
                    torch.testing.assert_close(input_state,before,rtol=0,atol=0)
                    torch.testing.assert_close(y,yy,atol=2e-3 if dtype==torch.bfloat16 else 2e-4,
                                               rtol=2e-2 if dtype==torch.bfloat16 else 2e-4)
                    torch.testing.assert_close(observed,expected,atol=2e-4,rtol=2e-3)
                    oe=max(oe,float((y.float()-yy.float()).abs().max()))
                    se=max(se,float((observed-expected).abs().max()))
                for final in [True,False]:
                    yy,ss=reference(q,k,v,g,beta,None,final,True)
                    y,s=adapter(q,k,v,g,beta,None,final,True)
                    torch.testing.assert_close(y,yy,atol=2e-3,rtol=2e-2)
                    if final:torch.testing.assert_close(s,ss,atol=2e-4,rtol=2e-3)
                    else:assert s is None
                args=(q,k,v,g,beta,h,True,True)
                base=timed(reference,args);opt=timed(adapter,args)
                rows.append(dict(dtype=str(dtype),batch=batch,heads=heads,kdim=kdim,vdim=vdim,
                    fuse_hf_norm=fuse,max_output_error=oe,max_state_error=se,reference_ms=base,
                    adapter_ms=opt,speedup=base/opt,input_state_preserved=True))
    return dict(status='PASS',cases=rows)
