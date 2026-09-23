"""Residual-only Write/Read pilot for Qwen3.5/3.8 text towers.

Document states are independent chunks. Lower-layer online state contains only
the sink/query/generated sequence (sink is independently written), while upper
layers consume the assembled residual pack. Two transient caches retain the
native attention KV and DeltaNet recurrent/conv state in their respective bands.
Neither cache is a persistent document memory. All quality paths share selection.
"""
import contextlib
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache


class LoRALinear(nn.Module):
    def __init__(self, base, rank=32, alpha=32):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.enabled = True
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + ((x.float() @ self.a.T) @ self.b.T).to(y.dtype) * self.scale
        return y


class HybridReader:
    def __init__(self, model, j):
        self.model = model
        self.core = getattr(model.model, 'language_model', model.model)
        self.config = self.core.config
        self.j = j
        self.L = len(self.core.layers)
        assert self.config.model_type == 'qwen3_5_text'
        assert 0 < j < self.L
        self.device = self.core.embed_tokens.weight.device
        self.modules = {}
        model.requires_grad_(False)

    def attach(self):
        for i in range(self.j, self.L):
            block = self.core.layers[i]
            for name, module in list(block.named_modules()):
                if isinstance(module, nn.Linear):
                    parent, _, child = name.rpartition('.')
                    wrapped = LoRALinear(module)
                    setattr(block.get_submodule(parent) if parent else block, child, wrapped)
                    self.modules[f'layers.{i}.{name}'] = wrapped
        return [p for mod in self.modules.values() for p in (mod.a, mod.b)]

    @contextlib.contextmanager
    def adapter(self, enabled):
        old = [m.enabled for m in self.modules.values()]
        for m in self.modules.values():
            m.enabled = enabled
        try:
            yield
        finally:
            for m, state in zip(self.modules.values(), old):
                m.enabled = state

    def tensor(self, ids):
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def layers(self, hidden, start, end, cache=None, offset=0, grad=False):
        positions = torch.arange(offset, offset + hidden.shape[1], device=self.device)[None]
        positions = positions.expand(hidden.shape[0], -1)
        rotary = self.core.rotary_emb(hidden, positions[None].expand(3, -1, -1))
        # No padding: SDPA uses causal attention on a multi-token prefill and
        # unrestricted access to cached history for a one-token decode. Native
        # DeltaNet is causal without an additive attention mask.
        for block in self.core.layers[start:end]:
            def call(h, layer=block):
                return layer(h, position_embeddings=rotary, attention_mask=None,
                             position_ids=positions, past_key_values=cache,
                             use_cache=cache is not None)
            hidden = checkpoint(call, hidden, use_reentrant=False) if grad else call(hidden)
        return hidden

    @torch.no_grad()
    def write(self, ids):
        return self.layers(self.core.embed_tokens(self.tensor(ids)), 0, self.j).detach()

    def logits(self, hidden, last=1):
        return self.model.lm_head(self.core.norm(hidden[:, -last:]))

    def full_hidden(self, ids):
        return self.layers(self.core.embed_tokens(self.tensor(ids)), 0, self.L)

    def cache_hidden(self, segments, grad=False):
        h = torch.cat([self.write(ids) for ids in segments], dim=1)
        return self.layers(h, self.j, self.L, grad=grad)

    @torch.no_grad()
    def generate(self, segments, mode, max_tokens, eos_ids, use_cache=True):
        assert mode in ('replay', 'cache')
        whole = sum(segments, [])
        generated = []
        if not use_cache:
            fixed = torch.cat([self.write(ids) for ids in segments[:-1]], dim=1) if mode == 'cache' else None
            for _ in range(max_tokens):
                if mode == 'replay':
                    hidden = self.full_hidden(whole + generated)
                else:
                    query = self.write(segments[-1] + generated)
                    hidden = self.layers(torch.cat([fixed, query], dim=1), self.j, self.L)
                token = int(self.logits(hidden).argmax(-1).item())
                generated.append(token)
                if token in eos_ids:
                    break
            return generated
        if mode == 'replay':
            upper = DynamicCache(config=self.config)
            hidden = self.layers(self.core.embed_tokens(self.tensor(whole)), 0, self.L, cache=upper)
            upper_position = len(whole)
        else:
            lower, upper = DynamicCache(config=self.config), DynamicCache(config=self.config)
            fixed = torch.cat([self.write(ids) for ids in segments[:-1]], dim=1)
            query = self.layers(self.core.embed_tokens(self.tensor(segments[-1])), 0, self.j, cache=lower)
            hidden = self.layers(torch.cat([fixed, query], dim=1), self.j, self.L, cache=upper)
            lower_position = len(segments[-1])
            upper_position = len(whole)
        for step in range(max_tokens):
            token = int(self.logits(hidden).argmax(-1).item())
            generated.append(token)
            if token in eos_ids or step == max_tokens - 1:
                break
            hidden = self.core.embed_tokens(self.tensor([token]))
            if mode == 'replay':
                hidden = self.layers(hidden, 0, self.L, cache=upper, offset=upper_position)
            else:
                hidden = self.layers(hidden, 0, self.j, cache=lower, offset=lower_position)
                hidden = self.layers(hidden, self.j, self.L, cache=upper, offset=upper_position)
                lower_position += 1
            upper_position += 1
        return generated

    @torch.no_grad()
    def validate(self, tokenizer):
        ids = tokenizer.encode('Alice was born in Paris and Bob was born in Rome. Alice lives in London. Where was Alice born?', add_special_tokens=False)
        stock = self.model(input_ids=self.tensor(ids), use_cache=False, logits_to_keep=1).logits
        manual = self.logits(self.full_hidden(ids))
        continuous = self.logits(self.layers(self.write(ids), self.j, self.L))
        checks = {'stock_max_abs': float((stock-manual).abs().max()),
                  'continuous_max_abs': float((stock-continuous).abs().max())}
        assert torch.allclose(stock, manual, atol=2e-3, rtol=2e-3), checks
        assert torch.allclose(stock, continuous, atol=2e-3, rtol=2e-3), checks
        segments = [ids[:1], ids[1:17], ids[17:]]
        for mode in ('replay', 'cache'):
            cached = self.generate(segments, mode, 5, set(), True)
            recomputed = self.generate(segments, mode, 5, set(), False)
            checks[mode] = {'cached': cached, 'recomputed': recomputed, 'equal': cached == recomputed}
            assert cached == recomputed, checks
        return checks

    def save(self, path):
        torch.save({'j': self.j, 'rank': 32, 'alpha': 32,
                    'modules': {k: {'a': m.a.detach().cpu(), 'b': m.b.detach().cpu()} for k,m in self.modules.items()}}, path)
