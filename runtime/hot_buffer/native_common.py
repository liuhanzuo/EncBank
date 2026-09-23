"""Shared fixed settings for the full newer-backbone benchmark extension."""
import json, os, platform, random
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).resolve().parent
REMOTE = '/srv/encbank/comem_new_backbones_formal_20260915'
MODELS = [
    dict(name='Qwen3.5-9B', j=6, L=32, revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'),
    dict(name='Qwen3.8-27B', j=21, L=64, revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'),
]
for cfg in MODELS:
    cfg['path'] = '/srv/encbank/comem_new_backbones_20260915/models/' + cfg['name']
TRAIN_DATA = '/srv/encbank/comem_new_backbones_20260915/data/pg19_train_64.jsonl'
ARMS = ['cache_lora', 'cache_without_lora', 'replay_base', 'replay_shared_lora',
        'kvdirect', 'streamingllm', 'hcache_style']
STEPS, WINDOW, CHUNK, SHARDS = 4000, 4096, 512, 4

def dump(path, value):
    import errno, sys, time
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    tmp = path.with_name(path.name + '.tmp')
    for attempt, delay in enumerate((0, 1, 2, 4, 8)):
        if delay: time.sleep(delay)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_text(encoding='utf8') == payload: return
                raise FileExistsError('Refuse to replace a different completed artifact: ' + str(path))
            with tmp.open('w', encoding='utf8') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            tmp.replace(path)
            return
        except OSError as exc:
            if exc.errno not in (errno.EIO, getattr(errno, 'EREMOTEIO', 121)) or attempt == 4: raise
            print(json.dumps(dict(event='same_payload_persistence_retry', path=str(path),
                                  errno=exc.errno, attempt=attempt + 1)), file=sys.stderr, flush=True)


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)

def seed_all():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(2)

def tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg['path'], local_files_only=True)
    tok.model_max_length = 10**9
    tok.pad_token = tok.eos_token
    return tok

def load_model(cfg):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    config = AutoConfig.from_pretrained(cfg['path'], local_files_only=True)
    cls = AutoModelForImageTextToText if hasattr(config, 'text_config') else AutoModelForCausalLM
    model = cls.from_pretrained(cfg['path'], dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa', device_map='cuda').eval()
    assert (getattr(config, 'text_config', config)).num_hidden_layers == cfg['L']
    return model

def metadata(cfg):
    dev = torch.cuda.get_device_properties(0)
    return dict(model=cfg, gpu=dev.name, gpu_memory_gib=dev.total_memory / 2**30,
                host=platform.node(), job=os.environ['SLURM_JOB_ID'],
                torch=torch.__version__, transformers=transformers.__version__,
                timing_is_infrastructure_result=False)

def flat_state(reader):
    return {k: {ab: getattr(m, ab).detach().cpu().clone() for ab in ('a', 'b')}
            for k, m in reader.modules.items()}

def load_state(reader, saved, cfg):
    assert saved['j'] == cfg['j'] and saved['model'] == cfg
    assert set(reader.modules) == set(saved['modules'])
    with torch.no_grad():
        for k, m in reader.modules.items():
            for ab in ('a', 'b'):
                src, dst = saved['modules'][k][ab], getattr(m, ab)
                assert src.shape == dst.shape
                dst.copy_(src)
