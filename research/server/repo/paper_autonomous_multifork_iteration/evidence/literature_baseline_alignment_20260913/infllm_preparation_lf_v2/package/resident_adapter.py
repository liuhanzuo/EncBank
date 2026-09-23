"""Prepared InfLLM/Qwen3 bridge. No loading, scoring, or GPU launch at import.

The upstream ContextManager, representative selection, local/global masks, RoPE,
and Torch joint-softmax implementation are loaded unchanged from the pinned tree.
Only MemoryUnit's storage policy changes. This is not an SDPA implementation.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

UPSTREAM_COMMIT = '12b70798f56e56ebb23c53c7018091a3f540a028'
ATTENTION_CONFIG = dict(n_init=128, n_local=4096, block_size=128,
    max_cached_block=32, topk=16, exc_block_size=512, repr_topk=4,
    fattn=False, cache_strategy='lru', score_decay=None, chunk_topk_calc=None,
    async_global_stream=False, pin_memory=False, faiss=False, perhead=False)


def load_upstream(source_root, lock_path):
    """Load only attention modules; never run old HF patch_hf or chat wrappers."""
    source_root = Path(source_root).resolve()
    lock = json.loads(Path(lock_path).read_text(encoding='utf-8'))
    assert lock['commit'] == UPSTREAM_COMMIT
    for rec in lock['files']:
        relative = rec['path'].split('/public_upstream/', 1)[1]
        path = source_root / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['sha256'], relative
    namespace = '_pinned_infllm'
    for name, directory in [(namespace, source_root/'inf_llm'),
                            (namespace+'.attention', source_root/'inf_llm/attention')]:
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        sys.modules[name] = module
    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, source_root/relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    cm = load(namespace+'.attention.context_manager', 'inf_llm/attention/context_manager.py')
    rope = load(namespace+'.attention.rope', 'inf_llm/attention/rope.py')
    return cm, rope.RotaryEmbeddingESM


def resident_memory_unit_class(torch, *, cpu_test=False):
    class ResidentMemoryUnit:
        """Same block values with permanent device backing; LRU is metadata only.

        The caller is synchronous (async_global_stream=False), so CUDA events and
        a second cached copy are unnecessary. The existing upstream CudaCache
        allocation is retained and reported, even though these units do not use it.
        No CPU K/V, pinned memory, disk, faiss, or runtime offload is used.
        """
        def __init__(self, kv, cache, load_to_cache=False, pin_memory=False):
            assert not pin_memory
            assert all(t.device == kv[0].device for t in kv)
            assert kv[0].device.type == ('cpu' if cpu_test else 'cuda')
            # stack owns only this block's storage, never a view of the remainder.
            self.resident_data = torch.stack(kv, dim=0).contiguous()
            self.active = bool(load_to_cache)

        def load(self, target=None):
            newly_active = not self.active
            self.active = True
            if target is not None:
                target[0].copy_(self.resident_data[0])
                target[1].copy_(self.resident_data[1])
            return newly_active, None

        def get(self):
            assert self.active
            return self.resident_data

        def offload(self):
            # Upstream name means cache eviction. Backing remains on the device.
            assert self.active
            self.active = False
    return ResidentMemoryUnit


def tensor_items(value, prefix='state', seen=None):
    import torch
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensor_items(child, f'{prefix}.{key}', seen)
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from tensor_items(child, f'{prefix}.{i}', seen)
    elif hasattr(value, '__dict__') and not isinstance(value, (type, types.ModuleType)):
        for key, child in vars(value).items():
            yield from tensor_items(child, f'{prefix}.{key}', seen)


class Entry:
    def __init__(self, states, document_id, prefix_tokens):
        self.states = states
        self.document_id = document_id
        self.prefix_tokens = prefix_tokens

    def tensor_items(self):
        return tensor_items(self.states)

    def inventory(self):
        tensors = list(self.tensor_items())
        allocations = {}
        for name, tensor in tensors:
            assert tensor.device.type == 'cuda'
            storage = tensor.untyped_storage()
            allocations[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        return {'unique_storage_bytes': sum(allocations.values()),
                'tensor_objects': len(tensors), 'unique_storages': len(allocations),
                'all_tensors_cuda': True, 'runtime_offload': False,
                'global_blocks_per_layer': [s.num_global_block for s in self.states],
                'scope': 'Actual persistent InfLLM tensors including retained upstream pool, local/remainder storage and representative keys; excludes allocator reservation'}

    def release(self):
        self.states = None


class Method:
    """Batch-one Qwen3 inference using native modules plus upstream InfLLM attention.

    Explicit residual/norm/MLP order matches transformers 5.5.4 Qwen3DecoderLayer.
    There is no HF DynamicCache, dense attention call, truncation, adapter, or H.
    """
    def __init__(self, model, tokenizer=None, *, upstream_root, upstream_lock,
                 attention_config=None, write_chunk_size=2048):
        import torch
        self.model = model
        self.bos = 151643
        self.memory = None
        self.configuration = dict(ATTENTION_CONFIG if attention_config is None else attention_config)
        assert not any(self.configuration[k] for k in ('fattn','async_global_stream','pin_memory','faiss','perhead'))
        assert self.configuration['cache_strategy'] == 'lru'
        assert model.config.model_type == 'qwen3'
        assert not any('lora_' in n for n, _ in model.named_parameters())
        assert not getattr(model, 'peft_config', None)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert all(p.device.type == 'cuda' and p.dtype == torch.float16 for p in model.parameters())
        assert not any(hasattr(m, '_hf_hook') for m in model.modules())
        assert not getattr(model, 'hf_device_map', None)
        assert all(getattr(layer.self_attn, 'sliding_window', None) is None for layer in model.model.layers)
        self.cm, Rope = load_upstream(upstream_root, upstream_lock)
        self.cm.MemoryUnit = resident_memory_unit_class(torch)
        rope_parameters = getattr(model.config, 'rope_parameters', None) or {}
        rope_type = rope_parameters.get('rope_type', 'default')
        assert rope_type == 'default', 'Scaled RoPE needs a separately specified InfLLM extension'
        base = rope_parameters.get('rope_theta', getattr(model.config, 'rope_theta', None))
        assert base is not None
        self.rope_base = base
        self.rope = Rope(model.config.head_dim, base=base, distance_scale=1.0)
        self.write_chunk_size = int(write_chunk_size)
        assert self.write_chunk_size > 0

    def _new_states(self):
        rope = copy.deepcopy(self.rope)  # Entry owns its rotary tensors for release checks.
        return [self.cm.ContextManager(rope, **self.configuration)
                for _ in self.model.model.layers]

    def forward(self, ids, states, *, emit_head):
        import torch
        device = self.model.model.embed_tokens.weight.device
        hidden = self.model.model.embed_tokens(torch.tensor([ids], device=device, dtype=torch.long))
        for layer, state in zip(self.model.model.layers, states):
            residual = hidden
            normed = layer.input_layernorm(hidden)
            attention = layer.self_attn
            shape = (*normed.shape[:-1], -1, attention.head_dim)
            q = attention.q_norm(attention.q_proj(normed).view(shape)).transpose(1, 2).contiguous()
            k = attention.k_norm(attention.k_proj(normed).view(shape)).transpose(1, 2).contiguous()
            v = attention.v_proj(normed).view(shape).transpose(1, 2).contiguous()
            output = state.append(q, k, v, q, k, v)
            output = output.transpose(1, 2).reshape(*normed.shape[:-1], -1).contiguous()
            hidden = residual + attention.o_proj(output)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.model.lm_head(self.model.model.norm(hidden[:, -1:])) if emit_head else None

    def write(self, document_ids, document_id):
        states = self._new_states()
        ids = [self.bos] + list(document_ids)
        for start in range(0, len(ids), self.write_chunk_size):
            self.forward(ids[start:start+self.write_chunk_size], states, emit_head=False)
        return Entry(states, document_id, len(ids))

    def open_request(self, entry, bare_question_ids):
        # Deep-copy stays on GPU and preserves immutable document reuse semantics.
        return Request(self, copy.deepcopy(entry.states), entry.prefix_tokens)

    def close(self):
        self.model = None
        self.rope = None


class Request:
    def __init__(self, method, states, prefix_tokens):
        self.method = method
        self.states = states
        self.prefix_tokens = prefix_tokens
        self.selected_indices = None  # Selection is per layer and per attention chunk.
        self.query_calls = self.decode_calls = self.head_calls = 0

    def tensor_items(self):
        return tensor_items(self.states)

    def consume(self, token, *, emit_head, phase):
        assert phase in ('query', 'decode')
        if phase == 'query':
            self.query_calls += 1
        else:
            self.decode_calls += 1
        self.head_calls += int(emit_head)
        return self.method.forward([token], self.states, emit_head=emit_head)

    def close(self):
        self.states = None
        self.method = None
