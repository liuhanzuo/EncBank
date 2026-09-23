"""PEFT-format CoMem adapters with an explicit legacy FP32 arithmetic option.

Model/adapter artifacts are written only below /srv/encbank.
Default loading retains the original operation order; merging is opt-in.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, PeftModel
from peft.tuners.lora.layer import Linear as PeftLinear
from safetensors.torch import load_file, save_file


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def export_adapter(saved, directory, core_prefix, model_path, revision):
    """Translate original a/b tensors, preserving all FP32 values and targets."""
    directory = Path(directory).resolve()
    directory.relative_to(Path('/srv/encbank').resolve())
    directory.mkdir(exist_ok=False)
    state = {}
    targets = []
    modules = saved['modules']
    assert modules and core_prefix
    for name, values in sorted(modules.items()):
        parts = name.split('.')
        assert parts[0] == 'layers' and int(parts[1]) >= saved['j'], name
        assert set(values) == {'a', 'b'}, name
        a, b = values['a'], values['b']
        assert a.ndim == b.ndim == 2 and a.shape[0] == b.shape[1] == 32
        assert a.dtype == b.dtype == torch.float32, name
        assert bool(torch.isfinite(a).all() and torch.isfinite(b).all()), name
        full_name = core_prefix + '.' + name
        targets.append(full_name)
        for source, dest in [('a', 'lora_A'), ('b', 'lora_B')]:
            state[f'base_model.model.{full_name}.{dest}.weight'] = values[source].detach().cpu().contiguous()
    config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, target_modules=targets,
                        bias='none', inference_mode=True, task_type=None,
                        base_model_name_or_path=model_path, revision=revision)
    config.save_pretrained(directory)
    save_file(state, str(directory / 'adapter_model.safetensors'))
    restored = load_file(str(directory / 'adapter_model.safetensors'))
    assert restored.keys() == state.keys()
    assert all(torch.equal(state[k], restored[k]) for k in state)
    receipt = dict(format='PEFT', module_count=len(targets), tensor_count=len(state),
                   rank=32, alpha=32, dtype='float32', j=saved['j'],
                   target_modules=targets, roundtrip_exact=True,
                   files={n: file_sha(directory / n) for n in ['adapter_config.json', 'adapter_model.safetensors']})
    (directory / 'conversion_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


class LegacyArithmetic(nn.Module):
    """Use PEFT-owned weights but retain the trained runner's rounding points."""
    def __init__(self, layer):
        super().__init__()
        assert isinstance(layer, PeftLinear)
        assert list(layer.active_adapters) == ['default'] and not layer.merged
        assert layer.scaling['default'] == 1.0
        assert layer.lora_A['default'].weight.dtype == torch.float32
        assert layer.lora_B['default'].weight.dtype == torch.float32
        self.peft_layer = layer

    def forward(self, x):
        layer = self.peft_layer
        assert not layer.merged and not layer.disable_adapters
        y = layer.base_layer(x)
        delta = (x.float() @ layer.lora_A['default'].weight.T) @ layer.lora_B['default'].weight.T
        return y + delta.to(y.dtype) * layer.scaling['default']


def load_comem_peft(model, directory, mode='compat'):
    """Load into a fresh raw model before creating its HybridReader.

compat: preserve original FP32 branch rounding. native: unmodified PEFT.
merged_bf16: standard PEFT merge_and_unload; numerical validation required.
Never call this on a model already wrapped by HybridReader.attach().
"""
    assert mode in {'compat', 'native', 'merged_bf16'}
    directory = Path(directory)
    receipt = json.loads((directory / 'conversion_receipt.json').read_text())
    for name, digest in receipt['files'].items():
        assert file_sha(directory / name) == digest, name
    peft_model = PeftModel.from_pretrained(model, str(directory), is_trainable=False,
        autocast_adapter_dtype=True, local_files_only=True)
    found = {name: layer for name, layer in model.named_modules() if isinstance(layer, PeftLinear)}
    assert set(found) == set(receipt['target_modules']), 'Missing or extra LoRA targets'
    for layer in found.values():
        assert layer.lora_A['default'].weight.dtype == torch.float32
        assert layer.lora_B['default'].weight.dtype == torch.float32
    if mode == 'merged_bf16':
        model = peft_model.merge_and_unload(safe_merge=True)
    elif mode == 'compat':
        for name, layer in found.items():
            parent, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, child, LegacyArithmetic(layer))
    model.requires_grad_(False)
    return model, peft_model, found
