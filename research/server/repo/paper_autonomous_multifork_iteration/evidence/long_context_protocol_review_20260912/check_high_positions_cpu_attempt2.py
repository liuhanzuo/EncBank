"""Targeted CPU RoPE plus direct native SDPA; no model construction or weights."""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
import torch
import transformers
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb
from transformers.integrations.sdpa_attention import sdpa_attention_forward

HERE=Path(__file__).resolve().parent
OUT=HERE/'cpu_high_position_report_attempt2.json'
EXPECTED={
    'models/qwen3/modeling_qwen3.py':'fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab',
    'integrations/sdpa_attention.py':'87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564',
    'modeling_rope_utils.py':'55edda248757ae2ab38ee35bb79a4cb97660e31e40d95dc421e3863e43097930',
}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@torch.inference_mode()
def main():
    assert not OUT.exists()
    report={'status':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'python':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,
            'scope':'CPU exact native Qwen RoPE/apply-rotation/native SDPA, tiny tensors only. No Qwen model construction, real weights, benchmark, GPU, full-length attention or resource/quality claim.',
            'checks':[],'cases':[],'tolerance':{'atol':.005,'rtol':.01},'script_sha256':sha(__file__)}
    def save():OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def check(name,ok,detail=None):
        report['checks'].append({'name':name,'passed':bool(ok),'detail':detail});save()
        assert ok,name
    save()
    try:
        check('exact framework and no CUDA initialization',transformers.__version__=='5.5.4' and torch.__version__.startswith('2.10.0') and not torch.cuda.is_initialized())
        report['runtime_source_sha256']={name:sha(Path(transformers.__file__).parent/name) for name in EXPECTED}
        check('exact frozen model SDPA and official RoPE files',report['runtime_source_sha256']==EXPECTED)
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20260912)
        cfg=Qwen3Config(hidden_size=4096,num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=40960,rope_theta=1000000,rope_scaling=None,use_sliding_window=False)
        cfg._attn_implementation='sdpa'
        rope=Qwen3RotaryEmbedding(cfg)
        report['effective_rope_parameters']=cfg.rope_parameters
        check('unscaled default 40960 metadata preserved',cfg.max_position_embeddings==40960 and rope.rope_type=='default' and rope.attention_scaling==1.0 and cfg.rope_parameters['rope_theta']==1000000)
        module=SimpleNamespace(num_key_value_groups=4,is_causal=True,training=False,config=cfg,layer_idx=0)
        q=torch.randn((1,32,4,128),dtype=torch.float32).half()
        k=torch.randn((1,8,4,128),dtype=torch.float32).half()
        v=torch.randn((1,8,4,128),dtype=torch.float32).half()
        like=torch.zeros((1,4,4096),dtype=torch.float16)
        for offset in (0,32768,40960,65536,131072,133120,262144):
            pos=torch.arange(4).unsqueeze(0)+offset
            cos,sin=rope(like,pos)
            freq=pos.float().unsqueeze(-1)*rope.inv_freq.reshape(1,1,-1)
            emb=torch.cat((freq,freq),dim=-1)
            qr,kr=apply_rotary_pos_emb(q,k,cos,sin)
            full,_=sdpa_attention_forward(module,qr,kr,v,attention_mask=None,dropout=0.,scaling=128**-.5,is_causal=True)
            cached,_=sdpa_attention_forward(module,qr[:,:,-1:],kr,v,attention_mask=None,dropout=0.,scaling=128**-.5,is_causal=False)
            kk=kr.repeat_interleave(4,dim=1).float();vv=v.repeat_interleave(4,dim=1).float()
            weights=qr.float()@kk.transpose(-1,-2)*(128**-.5)
            weights=weights.masked_fill(torch.ones(4,4,dtype=torch.bool).triu(1),float('-inf'))
            reference=(torch.softmax(weights,dim=-1)@vv).transpose(1,2).half()
            delta=float((full.float()-reference.float()).abs().max())
            cached_delta=float((full[:,-1:].float()-cached.float()).abs().max())
            check(f'offset{offset} direct FP32 phase oracle and finite rotation',torch.equal(cos,emb.cos().half()) and torch.equal(sin,emb.sin().half()) and torch.isfinite(qr).all() and torch.isfinite(kr).all())
            check(f'offset{offset} native FP16 SDPA versus independent FP32 causal attention and cached final query',torch.allclose(full,reference,atol=.005,rtol=.01) and torch.allclose(full[:,-1:],cached,atol=.005,rtol=.01) and torch.isfinite(full).all(),{'maxabs_native_reference':delta,'maxabs_full_cached':cached_delta})
            if offset>=40960:
                wrapc,wraps=rope(like,pos%40960)
                check(f'offset{offset} no position wrap',not torch.equal(cos,wrapc) and not torch.equal(sin,wraps))
            report['cases'].append({'offset':offset,'position_ids':pos.tolist(),'q_shape':list(q.shape),'k_shape':list(k.shape),'maxabs_native_reference':delta,'maxabs_full_cached':cached_delta})
        check('all probe tensors CPU and no CUDA context',all(t.device.type=='cpu' for t in (q,k,v,cos,sin,qr,kr,full,cached)) and not torch.cuda.is_initialized())
        report.update(status='PASS_CPU_HIGH_POSITION_ROPE_AND_DIRECT_NATIVE_SDPA',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
        print(json.dumps({'status':report['status'],'checks':len(report['checks']),'output':str(OUT)}))
    except BaseException as exc:
        report.update(status='FAILED_CPU_INFRASTRUCTURE_OR_PROBE_NOT_SCIENTIFIC_RESULT',error={'type':type(exc).__name__,'message':str(exc)});save();raise

if __name__=='__main__':main()
