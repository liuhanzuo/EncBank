"""Bounded CPU high-position probe; no checkpoint, benchmark, CUDA or training."""
import copy
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

HERE = Path(__file__).resolve().parent
OUT = HERE / 'cpu_high_position_report.json'
EXPECTED = {
    'models/qwen3/modeling_qwen3.py': 'fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab',
    'integrations/sdpa_attention.py': '87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564',
    'modeling_rope_utils.py': '55edda248757ae2ab38ee35bb79a4cb97660e31e40d95dc421e3863e43097930',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@torch.inference_mode()
def main():
    assert not OUT.exists(), 'Do not overwrite an existing probe result'
    report = {
        'status': 'running', 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python': sys.executable, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'scope': 'Tiny random CPU model and isolated rotary embedding only; no full-length attention, model weights, task data, quality, capacity or GPU qualification.',
        'tolerance': {'atol': 0.005, 'rtol': 0.01}, 'checks': [], 'cases': [],
        'script_sha256': sha(__file__),
    }
    def save():
        OUT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    def check(name, value, detail=None):
        report['checks'].append({'name': name, 'passed': bool(value), 'detail': detail})
        save()
        assert value, name
    save()
    try:
        check('registered exact framework versions', transformers.__version__ == '5.5.4' and torch.__version__.startswith('2.10.0'))
        report['runtime_source_sha256'] = {name: sha(Path(transformers.__file__).parent/name) for name in EXPECTED}
        check('exact frozen native SDPA/model and official RoPE source hashes', report['runtime_source_sha256'] == EXPECTED)
        check('CUDA remains uninitialized', not torch.cuda.is_initialized())
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20260912)
        cfg = Qwen3Config(vocab_size=127, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=128, max_position_embeddings=40960, rope_theta=1000000,
                           rope_scaling=None, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                           attention_dropout=0.0, use_sliding_window=False)
        report['effective_rope_parameters'] = cfg.rope_parameters
        check('unchanged default position metadata and unscaled RoPE', cfg.max_position_embeddings == 40960 and cfg.rope_parameters == {'rope_type': 'default', 'rope_theta': 1000000})
        rope = Qwen3RotaryEmbedding(cfg)
        positions = torch.tensor([[0,32767,32768,40959,40960,65535,65536,131071,131072,133119,262144]], dtype=torch.long)
        x = torch.zeros((1,positions.numel(),128), dtype=torch.float16)
        cos, sin = rope(x, positions)
        freq = positions.float().unsqueeze(-1) * rope.inv_freq.reshape(1,1,-1)
        emb = torch.cat((freq,freq), dim=-1)
        check('FP16 RoPE at exact high positions equals direct FP32 trigonometric oracle', torch.equal(cos,emb.cos().half()) and torch.equal(sin,emb.sin().half()) and torch.isfinite(cos).all() and torch.isfinite(sin).all(), positions.tolist())
        wrapped_cos, wrapped_sin = rope(x, positions % 40960)
        check('high positions are not wrapped to config metadata', not torch.equal(cos,wrapped_cos) and not torch.equal(sin,wrapped_sin))
        metadata_only = copy.deepcopy(cfg)
        metadata_only.max_position_embeddings = 262145
        metadata_rope = Qwen3RotaryEmbedding(metadata_only)
        check('changing default max_position metadata alone does not change frequencies', torch.equal(rope.inv_freq,metadata_rope.inv_freq) and rope.attention_scaling == metadata_rope.attention_scaling == 1.0)
        cfg._attn_implementation = 'eager'
        eager = Qwen3ForCausalLM(cfg).to(dtype=torch.float16).eval()
        sdpa = copy.deepcopy(eager)
        sdpa.config._attn_implementation = 'sdpa'
        ids = torch.tensor([[10,11,12,13]],dtype=torch.long)
        for offset in (0,40960,65536,131072,262144):
            pos = torch.arange(4).unsqueeze(0)+offset
            e = eager(input_ids=ids,position_ids=pos,use_cache=False).logits
            s = sdpa(input_ids=ids,position_ids=pos,use_cache=False).logits
            pref = sdpa(input_ids=ids[:,:3],position_ids=pos[:,:3],use_cache=True)
            step = sdpa(input_ids=ids[:,3:],position_ids=pos[:,3:],past_key_values=pref.past_key_values,use_cache=True)
            delta = float((e.float()-s.float()).abs().max())
            cache_delta = float((s[:,-1:].float()-step.logits.float()).abs().max())
            check(f'offset{offset} native SDPA/eager and cached last-token agree', torch.isfinite(s).all() and torch.allclose(e,s,atol=.005,rtol=.01) and torch.allclose(s[:,-1:],step.logits,atol=.005,rtol=.01) and step.past_key_values.get_seq_length()==4, {'maxabs_eager_sdpa':delta,'maxabs_cached_full':cache_delta})
            report['cases'].append({'offset':offset,'position_ids':pos.tolist(),'cache_tokens':4,'maxabs_eager_sdpa':delta,'maxabs_cached_full':cache_delta})
        check('model parameters stay FP16 CPU; unchanged 40960 metadata; CUDA uninitialized', all(p.device.type=='cpu' and p.dtype==torch.float16 for p in sdpa.parameters()) and sdpa.config.max_position_embeddings==40960 and not torch.cuda.is_initialized())
        report['status'] = 'PASS_CPU_HIGH_POSITION_MATH_AND_SMALL_NATIVE_SDPA'
        report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        print(json.dumps({'status':report['status'],'checks':len(report['checks']),'output':str(OUT)}))
    except BaseException as exc:
        report.update(status='FAILED_CPU_PROBE_NOT_SCIENTIFIC_EVIDENCE',error={'type':type(exc).__name__,'message':str(exc)})
        save()
        raise

if __name__ == '__main__':
    main()
