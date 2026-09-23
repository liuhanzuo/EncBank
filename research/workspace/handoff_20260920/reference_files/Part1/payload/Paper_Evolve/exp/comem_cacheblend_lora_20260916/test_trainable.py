"""Tiny-model correctness and gradient checks, CPU only; no experiment scores."""
import json
import config
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from comem import CoMem
from kv_reference import KVControl,assemble
from trainable_cacheblend import TrainableBlend
from train_support import attach_lora

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg=Qwen3Config(vocab_size=127,hidden_size=48,intermediate_size=96,num_hidden_layers=5,
        num_attention_heads=4,num_key_value_heads=2,head_dim=12,max_position_embeddings=256,
        attention_dropout=0.,bos_token_id=1,eos_token_id=2)
    cfg._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(cfg).eval()
    modules=attach_lora(model,2,4,4,torch.float32)
    # Nonzero adapters ensure parity covers adapted cached K/V, not zero-init only.
    with torch.no_grad():
        for m in modules.values():m.B.normal_(std=.02)
    reader=TrainableBlend(model,2,True)
    segments=[[1],list(range(3,19)),list(range(21,37)),list(range(41,49))]
    pack=torch.tensor([sum(segments,[])])
    with torch.no_grad():
        h=reader.hidden(segments,1.)
        actual=model.lm_head(h)
        expected=model(pack,use_cache=False).logits[:,-8:]
        error=float((actual-expected).abs().max());assert error<2e-5,error
        ref=KVControl(CoMem(model,0))
        ids,merged=assemble(ref,segments[1:-1],segments[-1],1)
        a,_,_=ref.read(ids,merged,1,8,.15,logits_tail=None)
        b=model.lm_head(reader.hidden(segments,.15))
        parity=float((a[:,-8:]-b).abs().max());assert parity<2e-5,parity
    model.zero_grad(set_to_none=True)
    reader.hidden(segments,1.).square().mean().backward()
    full_grads={k:m.B.grad.clone() for k,m in modules.items()}
    model.zero_grad(set_to_none=True)
    full=model.model(pack,use_cache=False).last_hidden_state[:,-8:]
    full.square().mean().backward()
    grad_error=max(float((full_grads[k]-m.B.grad).abs().max()) for k,m in modules.items())
    assert grad_error<2e-5,grad_error
    model.zero_grad(set_to_none=True)
    ids,merged=reader.assemble(segments)
    merged[3][0].retain_grad();merged[3][1].retain_grad()
    h,ix=reader.read_hidden(ids,merged,1,8,.15)
    h[:,-1].sum().backward()
    omitted=torch.tensor([i for i in range(1,33) if i not in ix.tolist()])
    writer_grad=float(merged[3][1].grad[:,:,omitted].abs().sum())
    assert writer_grad>0, 'Independent cache writer gradient was lost'
    assert all(m.A.grad is not None and m.B.grad is not None for m in modules.values())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    checked={k:(m.A.grad.clone(),m.B.grad.clone()) for k,m in modules.items()}
    model.zero_grad(set_to_none=True)
    reader.grad_checkpoint=False
    reader.hidden(segments,.15)[:,-1].sum().backward()
    checkpoint_error=max(float((g-p.grad).abs().max()) for k,m in modules.items() for g,p in zip(checked[k],(m.A,m.B)))
    assert checkpoint_error<2e-5,checkpoint_error
    report=dict(passed=True,full_recompute_logits_error=error,reference_r015_error=parity,
        full_recompute_gradient_error=grad_error,checkpoint_gradient_error=checkpoint_error,
        independent_writer_gradient_nonzero=writer_grad,all_adapter_gradients_connected=True,
        backbone_frozen=True,note='CPU random tiny model, software checks only')
    (config.ROOT/'cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
