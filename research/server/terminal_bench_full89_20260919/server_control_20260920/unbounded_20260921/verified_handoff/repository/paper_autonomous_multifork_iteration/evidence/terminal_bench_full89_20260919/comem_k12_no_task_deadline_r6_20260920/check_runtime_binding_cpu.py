"""Offline regression check for the failed path assertion and strict checkpoint identity."""
from pathlib import Path
import ast, copy, datetime, hashlib, json, os
from unittest.mock import patch
from types import SimpleNamespace
import common
from runtime_config import bind_runtime_config

H = Path(__file__).resolve().parent
P = json.loads((H/'plan.json').read_text())
M = json.loads((H/'model_recovery_manifest.json').read_text())
cfg = common.MODELS[1]
original = copy.deepcopy(cfg)
runtime = bind_runtime_config(cfg, P, M)
assert cfg == original and runtime is not cfg
assert {k: v for k, v in runtime.items() if k != 'path'} == {k: v for k, v in cfg.items() if k != 'path'}
assert not common.torch.cuda.is_initialized()
config = common.AutoConfig.from_pretrained(runtime['path'], local_files_only=True)
assert getattr(config, 'text_config', config).num_hidden_layers == cfg['L']
tok = common.tokenizer(runtime)
assert tok.pad_token == tok.eos_token
class StubModel:
    def eval(self): return self
cls = common.AutoModelForImageTextToText if hasattr(config, 'text_config') else common.AutoModelForCausalLM
with patch.dict(os.environ, {'SLURM_JOB_ID': 'CPU_BINDING_CHECK_ONLY'}), patch.object(common.torch.cuda, 'device_count', return_value=1), patch.object(cls, 'from_pretrained', return_value=StubModel()) as loader:
    common.load_model(runtime)
    assert loader.call_args.args == (P['model'],)
    assert loader.call_args.kwargs == dict(dtype=common.torch.bfloat16, local_files_only=True, attn_implementation='sdpa', device_map='cuda')
cp = Path(P['adapter_path'])
assert hashlib.sha256(cp.read_bytes()).hexdigest() == P['adapter_sha256']
saved = common.torch.load(cp, map_location='cpu', weights_only=False)
assert saved['step'] == 4000 and saved['j'] == cfg['j'] and saved['model'] == cfg
copied = []
class StubDestination:
    def __init__(self, src): self.shape = src.shape
    def copy_(self, src):
        assert src.device.type == 'cpu' and src.shape == self.shape
        copied.append(tuple(src.shape))
reader = SimpleNamespace(modules={k: SimpleNamespace(**{ab: StubDestination(v[ab]) for ab in ('a', 'b')}) for k, v in saved['modules'].items()})
common.load_state(reader, saved, cfg)
assert len(copied) == 2 * len(saved['modules'])
try:
    common.load_state(reader, saved, runtime)
except AssertionError: pass
else: raise AssertionError('Checkpoint must reject changed training identity')
negative = []
for key, value in [('model_revision', 'wrong'), ('j', cfg['j']+1), ('adapter_sha256', 'wrong'), ('model', '/srv/encbank/wrong')]:
    bad = dict(P, **{key: value})
    try: bind_runtime_config(cfg, bad, M)
    except AssertionError: negative.append(key)
    else: raise AssertionError(key)
tree = ast.parse((H/'agent_worker.py').read_text())
calls = [(n.func.id, [ast.unparse(x) for x in n.args]) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
assert ('tokenizer', ['runtime_cfg']) in calls and ('load_model', ['runtime_cfg']) in calls
assert ('load_state', ['reader', 'ckpt', 'cfg']) in calls
assert cfg == original and not common.torch.cuda.is_initialized()
out = dict(status='PASS', at=datetime.datetime.now().astimezone().isoformat(),
    actual_tokenizer_path=P['model'], mocked_model_loader_path=loader.call_args.args[0],
    actual_config_layers=cfg['L'], checkpoint_model=saved['model'], checkpoint_j=saved['j'], checkpoint_step=saved['step'],
    checkpoint_modules=len(saved['modules']), strict_load_state_retained=True, identity_not_mutated=True,
    negative_checks=negative, cuda_initialized=False, allocated_gpu_jobs=0, model_forwards=0,
    plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),
    source_sha256={n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in ['runtime_config.py','check_runtime_binding_cpu.py','common.py','agent_worker.py']})
(H/'runtime_binding_cpu.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out))
