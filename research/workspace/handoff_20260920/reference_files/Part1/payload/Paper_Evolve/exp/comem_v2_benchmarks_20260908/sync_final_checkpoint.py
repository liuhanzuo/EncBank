"""Copy the completed remote LoRA adapter for local RTX 5090 timing.

The remote checkpoint is retained. No GPU is used by this transfer or validation.
Called by the existing result-sync process after reading remote training status.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = '/data/liuhanzuo/comem_v2_20260908/outputs/8b_j12_pub_4k'
LOCAL = HERE / 'checkpoints/8b_j12_pub_4k'
STATUS = HERE / 'results/checkpoint_transfer_status.json'
FILES = ('final/adapter.pt', 'final/adapter_model.safetensors',
         'final/adapter_config.json', 'status.json', 'metadata.json')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def validate_adapter(folder):
    import torch
    from safetensors import safe_open
    state = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
    if not state.get('complete') or state.get('step') != 4000 or state.get('target_steps') != 4000:
        raise ValueError('Only the completed 4000-step run qualifies for final timing')
    # This is our own training export, whose metadata includes TorchVersion.
    payload = torch.load(folder / 'final/adapter.pt', map_location='cpu', weights_only=False)
    recipe = payload['metadata']['recipe']
    if payload.get('step') != 4000 or any(recipe.get(k) != v for k, v in
            {'steps': 4000, 'j': 12, 'rank': 32, 'alpha': 32, 'loss': 'published',
             'adapter_dtype': 'float32'}.items()):
        raise ValueError('Transferred adapter step or training recipe differs')
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    # JSON normalizes integer keys in model-config dictionaries to strings.
    if metadata['recipe'] != json.loads(json.dumps(recipe, default=str)):
        raise ValueError('Adapter metadata does not match transferred metadata.json')
    config = json.loads((folder / 'final/adapter_config.json').read_text(encoding='utf-8'))
    if config.get('r') != 32 or config.get('lora_alpha') != 32:
        raise ValueError('PEFT config does not match the published-method recipe')
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    layers = list(range(12, recipe['model_config']['num_hidden_layers']))
    if (set(config.get('target_modules', [])) != set(targets)
            or config.get('layers_to_transform') != layers
            or config.get('lora_dropout') != 0.0 or config.get('bias') != 'none'):
        raise ValueError('PEFT target layers or modules differ from the training export')
    expected = {}
    for key, value in payload['named'].items():
        _, index, projection, branch = key.split('.')
        parent = 'self_attn' if projection in targets[:4] else 'mlp'
        expected[f'base_model.model.model.layers.{index}.{parent}.{projection}.lora_{branch}.weight'] = value
    with safe_open(folder / 'final/adapter_model.safetensors', framework='pt', device='cpu') as tensors:
        if not expected or set(tensors.keys()) != set(expected):
            raise ValueError('PEFT tensor keys differ from the completed adapter.pt export')
        for key, value in expected.items():
            actual = tensors.get_tensor(key)
            if actual.dtype != value.dtype or not torch.equal(actual, value):
                raise ValueError(f'PEFT tensor differs from completed adapter.pt: {key}')


def sync_final_checkpoint(training):
    now = datetime.now(timezone.utc).isoformat()
    marker = LOCAL / 'TRANSFER_COMPLETE.json'
    if marker.exists():
        ready = json.loads(marker.read_text(encoding='utf-8'))
        if ready.get('complete') and ready.get('step') == 4000 and all((LOCAL / f).is_file() for f in FILES):
            return ready
        raise ValueError('Existing checkpoint transfer marker is inconsistent')
    if not training.get('complete') or training.get('step') != 4000 or training.get('target_steps') != 4000:
        waiting = {'status': 'waiting_for_final_training', 'complete': False,
                   'observed_checkpoint_step': training.get('step'), 'required_step': 4000,
                   'updated_at': now, 'remote_copy_retained': True,
                   'target': str(LOCAL / 'final')}
        save_json(STATUS, waiting)
        return waiting
    incoming = LOCAL / '.incoming'
    (incoming / 'final').mkdir(parents=True, exist_ok=True)
    for relative in FILES:
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                        f'longjing-1:{REMOTE}/{relative}', str(incoming / relative)],
                       check=True, timeout=300)
    validate_adapter(incoming)
    (LOCAL / 'final').mkdir(parents=True, exist_ok=True)
    for relative in FILES:
        os.replace(incoming / relative, LOCAL / relative)
    ready = {'status': 'completed', 'complete': True, 'step': 4000,
             'adapter_dir': str((LOCAL / 'final').resolve()), 'adapter_load_mode': 'peft_unmerged',
             'required_timing_gpu': 'NVIDIA GeForce RTX 5090', 'remote_source': REMOTE,
             'remote_copy_retained': True, 'updated_at': datetime.now(timezone.utc).isoformat(),
             'files': {f: (LOCAL / f).stat().st_size for f in FILES}}
    save_json(marker, ready)
    save_json(STATUS, ready)
    return ready


if __name__ == '__main__':
    state = HERE / 'results/remote/outputs/8b_j12_pub_4k/status.json'
    print(json.dumps(sync_final_checkpoint(json.loads(state.read_text(encoding='utf-8')) if state.exists() else {})))
