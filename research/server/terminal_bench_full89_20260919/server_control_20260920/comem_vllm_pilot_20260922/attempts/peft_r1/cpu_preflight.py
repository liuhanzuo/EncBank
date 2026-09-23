"""Real PEFT save/load/merge and legacy arithmetic tests without model inference."""
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn
import peft
from hybrid_reader import LoRALinear
from comem_peft import export_adapter, load_comem_peft

H = Path(__file__).resolve().parent


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {'model_type': 'custom', 'tie_word_embeddings': False}
        self.core = nn.Module()
        self.core.layers = nn.ModuleList([nn.Linear(64, 64, bias=False).to(torch.bfloat16) for _ in range(3)])


def main():
    torch.set_num_threads(2)
    torch.manual_seed(91)
    base = Toy()
    original = copy.deepcopy(base)
    wrapped = LoRALinear(base.core.layers[2])
    with torch.no_grad():
        wrapped.b.normal_(std=0.05)
    saved = dict(j=2, modules={'layers.2': {'a': wrapped.a.detach().cpu(), 'b': wrapped.b.detach().cpu()}})
    exported = H / 'cpu_toy_adapter'
    receipt = export_adapter(saved, exported, 'core', 'synthetic-no-model', 'synthetic')
    outputs = {}
    x = torch.randn(2, 3, 64).to(torch.bfloat16)
    with torch.inference_mode():
        expected = wrapped(x)
        for mode in ['compat', 'native', 'merged_bf16']:
            model, owner, layers = load_comem_peft(copy.deepcopy(original), exported, mode)
            actual = model.core.layers[2](x)
            assert torch.equal(model.core.layers[0].weight, original.core.layers[0].weight)
            assert torch.equal(model.core.layers[1].weight, original.core.layers[1].weight)
            outputs[mode] = dict(max_abs=float((actual.float()-expected.float()).abs().max()),
                                 exact=torch.equal(actual, expected), finite=bool(torch.isfinite(actual).all()))
            assert outputs[mode]['finite']
        assert outputs['compat']['exact']
    result = dict(status='PASS', epoch=time.time(), peft=peft.__version__, torch=torch.__version__,
                  benchmark_attempts=0, model_calls=0, converted_tensor_roundtrip=receipt['roundtrip_exact'],
                  lower_layers_unchanged=True, modes=outputs)
    with (H/'cpu_preflight.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
