"""Synthetic state/output equivalence; no model weights or benchmark requests.

CPU mode checks the state-layout contract with an independent matrix oracle.
Interpreter mode runs the real installed Triton source on CPU, not compiled CUDA.
CUDA mode must run in a separate Slurm GPU allocation within the four-GPU cap.
"""
import argparse
import json
import os
import time


def reference(q,k,v,g,beta,state,normalize):
    import torch
    if normalize:
        q=q*(q.square().sum(-1,keepdim=True)+1e-6).rsqrt()
        k=k*(k.square().sum(-1,keepdim=True)+1e-6).rsqrt()
    q,k,v,g,beta=[t.float() for t in (q,k,v,g,beta)]
    b,_,h,key_dim=q.shape
    s=torch.zeros((b,h,key_dim,v.shape[-1]),device=q.device) if state is None else state.clone()
    s=s*g[:,0].exp()[...,None,None]
    delta=(v[:,0]-(s*k[:,0,...,None]).sum(-2))*beta[:,0,...,None]
    s=s+k[:,0,...,None]*delta[...,None,:]
    y=(s*(q[:,0]*key_dim**-.5)[...,None]).sum(-2)[:,None]
    return y,s


def matrix_oracle(q,k,v,g,beta,initial_state,inplace_final_state,use_qk_l2norm_in_kernel,ssm_state_indices):
    import torch
    q,k,v=q.float(),k.float(),v.float()
    if use_qk_l2norm_in_kernel:
        q=q*(q.square().sum(-1,keepdim=True)+1e-6).rsqrt()
        k=k*(k.square().sum(-1,keepdim=True)+1e-6).rsqrt()
    b,_,h,key_dim=q.shape
    assert (ssm_state_indices>0).all() and ssm_state_indices.unique().numel()==b
    s=initial_state[ssm_state_indices.long()].reshape(b*h,v.shape[-1],key_dim)
    s.mul_(g[:,0].exp().reshape(b*h,1,1))
    key=k[:,0].reshape(b*h,key_dim,1)
    residual=(v[:,0].reshape(b*h,v.shape[-1],1)-torch.bmm(s,key))*beta[:,0].reshape(b*h,1,1)
    s.add_(torch.bmm(residual,key.transpose(1,2)))
    y=torch.bmm(s,q[:,0].reshape(b*h,key_dim,1)*key_dim**-.5).reshape(b,1,h,v.shape[-1])
    initial_state[ssm_state_indices.long()]=s.reshape(b,h,v.shape[-1],key_dim)
    return y,initial_state


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','interpreter','cuda'],required=True)
    args=ap.parse_args()
    if args.backend=='interpreter':
        assert os.environ.get('TRITON_INTERPRET')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Set interpreter/CPU isolation before startup'
    import torch
    from vllm_gdn_adapter import VllmGdnDecode
    torch.set_num_threads(1);torch.manual_seed(4201)
    if args.backend=='cuda':
        assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    device='cuda' if args.backend=='cuda' else 'cpu'
    adapter=VllmGdnDecode(matrix_oracle if args.backend=='cpu' else None)
    cases=[];started=time.time()
    with torch.inference_mode():
        for batch,heads,kdim,vdim in [(1,2,16,8),(3,2,32,16),(8,48,128,128)]:
            for normalize in [False,True]:
                state=torch.randn(batch,heads,kdim,vdim,device=device)*.05
                expect=state.clone();observed=state.clone()
                max_output=max_state=0.
                for step in range(4):
                    q=torch.randn(batch,1,heads,kdim,device=device)*.1
                    k=torch.randn_like(q);v=torch.randn(batch,1,heads,vdim,device=device)
                    g=-torch.rand(batch,1,heads,device=device);beta=torch.rand_like(g)
                    input_state=observed;before=input_state.clone()
                    expected_y,expect=reference(q,k,v,g,beta,expect,normalize)
                    y,observed=adapter(q,k,v,g,beta,observed,True,normalize)
                    torch.testing.assert_close(input_state,before,rtol=0,atol=0)
                    torch.testing.assert_close(y,expected_y,atol=2e-4,rtol=2e-4)
                    torch.testing.assert_close(observed,expect,atol=2e-4,rtol=2e-4)
                    max_output=max(max_output,float((y-expected_y).abs().max()))
                    max_state=max(max_state,float((observed-expect).abs().max()))
                z,empty=adapter(q,k,v,g,beta,None,False,normalize)
                expected_z,_=reference(q,k,v,g,beta,None,normalize)
                torch.testing.assert_close(z,expected_z,atol=2e-4,rtol=2e-4);assert empty is None
                cases.append(dict(batch=batch,heads=heads,kdim=kdim,vdim=vdim,normalize=normalize,
                    steps=4,max_output_error=max_output,max_state_error=max_state))
        try:
            adapter(q.expand(-1,2,-1,-1),k.expand(-1,2,-1,-1),v.expand(-1,2,-1,-1),g,beta,None,True)
        except ValueError:pass
        else:raise AssertionError('Multi-token inputs were not rejected')
    print(json.dumps(dict(status='PASS',backend=args.backend,cases=cases,seconds=time.time()-started,
        gpu_kernel_executed=args.backend=='cuda',actual_vllm_source_executed=args.backend!='cpu',
        model_calls=0,benchmark_attempts=0,production_approved=False)))


if __name__=='__main__':main()
