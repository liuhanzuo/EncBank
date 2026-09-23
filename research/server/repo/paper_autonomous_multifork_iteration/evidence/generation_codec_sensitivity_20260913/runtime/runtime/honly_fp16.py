"""Independent-chunk H-only storage over pinned official Encbank primitives.

No document lower KV is retained. Lexical raw IDs are CPU input/index metadata;
hidden states, packed codes and model/request state stay on the model device.
The official Write/read/decode mathematics and EOS policy are unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import torch

from extracted_minmax import PackedTensor, quantize_tensor
from vendor_encbank.model import Encbank
from vendor_encbank import selectors

CHUNK_SIZE = 512
SELECTOR = 'iter_bm25'
TOPK = 12
ITER_HOP_TOPK = 4
ITER_ROUNDS = 0


def _cpu_ids(value, name: str) -> torch.Tensor:
    """Own integer input metadata; never accept a GPU tensor for host staging."""
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError(f'{name} must be CPU input token IDs; no GPU-to-host staging')
        if value.dtype != torch.long:
            raise ValueError(f'{name} tensor dtype must be int64')
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise ValueError(f'{name} must be one-dimensional')
        return value.detach().contiguous().clone()
    values = list(value)
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError(f'{name} must contain nonnegative integer token IDs')
    return torch.tensor(values, dtype=torch.long, device='cpu')


def _storage_key(t: torch.Tensor):
    store = t.untyped_storage()
    return str(t.device), store.data_ptr(), store.nbytes()


@dataclass
class HOnlyEntry:
    document_id: str
    raw_ids: torch.Tensor | None
    chunks: tuple[PackedTensor, ...]
    sink: PackedTensor | None
    resume_j: int
    bits: int
    group_size: int
    reader_binding: str
    bos_token_id: int
    eos_token_id: int | None
    device: torch.device
    dtype: torch.dtype
    released: bool = False

    def assert_live(self):
        if self.released or self.raw_ids is None or self.sink is None:
            raise RuntimeError('Document entry has been released')

    def raw_chunks(self) -> tuple[torch.Tensor, ...]:
        self.assert_live()
        return tuple(self.raw_ids.split(CHUNK_SIZE))

    def tensor_items(self) -> tuple[tuple[str, torch.Tensor], ...]:
        """All persistent tensor backing, without model weights or Read scratch."""
        if self.released:
            return ()
        out = [('raw_ids', self.raw_ids)]
        for index, packed in enumerate((*self.chunks, self.sink)):
            prefix = f'chunks.{index}' if index < len(self.chunks) else 'sink'
            for field in ('data', 'scales', 'biases'):
                value = getattr(packed, field)
                if value is not None:
                    out.append((prefix + '.' + field, value))
        return tuple(out)

    def inventory(self) -> dict:
        """Actual unique tensor bytes; Python object/allocator overhead is separate."""
        items = self.tensor_items()
        unique = {}
        for _, tensor in items:
            unique[_storage_key(tensor)] = tensor
        by_device = {}
        for key in unique:
            by_device[key[0]] = by_device.get(key[0], 0) + key[2]
        raw_bytes = 0 if self.raw_ids is None else self.raw_ids.untyped_storage().nbytes()
        chunk_bytes = sum(p.nbytes for p in self.chunks)
        sink_bytes = 0 if self.sink is None else self.sink.nbytes
        return {
            'unique_tensor_storage_bytes': sum(k[2] for k in unique),
            'tensor_storage_bytes_by_device': by_device,
            'raw_id_cpu_tensor_bytes': raw_bytes,
            'packed_chunk_H_bytes': chunk_bytes,
            'native_FP16_sink_bytes': sink_bytes,
            'document_lower_KV_bytes': 0,
            'chunk_count': len(self.chunks),
            'document_tokens': 0 if self.raw_ids is None else self.raw_ids.numel(),
            'chunk_token_lengths': [p.original_shape[1] for p in self.chunks],
            'bits': self.bits, 'group_size': self.group_size,
            'resume_j': self.resume_j, 'released': self.released,
            'model_state_included': False, 'query_state_included': False,
            'python_object_overhead_included': False,
            'lexical_index': 'CPU raw int64 token backing; no persistent derived BM25 index',
            'native_hidden_or_KV_host_offload': False,
        }

    def release(self):
        """Explicit terminal release only; Read never modifies persistent fields."""
        self.raw_ids = None
        self.chunks = ()
        self.sink = None
        self.released = True


@dataclass
class SelectedRead:
    selected_indices: tuple[int, ...]
    hidden: tuple[torch.Tensor, ...]
    sink_hidden: torch.Tensor | None
    released: bool = False

    def tensor_items(self):
        if self.released:
            return ()
        return tuple((f'selected.{i}', h) for i, h in zip(self.selected_indices, self.hidden)) + (('sink', self.sink_hidden),)

    def release(self):
        self.hidden = ()
        self.sink_hidden = None
        self.released = True


class HOnlyMemory:
    def __init__(self, model, tokenizer=None, *, resume_j: int = 12,
                 bits: int = 16, group_size: int = 64,
                 reader_binding: str = 'unbound-development',
                 bos_token_id: int | None = None,
                 eos_token_id: int | None = None):
        if (bits, group_size) not in ((16,64),(8,64),(4,64),(4,32),(4,128)):
            raise ValueError('Only the five frozen generation-codec arms are supported')
        # Peft.get_base_model preserves the installed active LoRA layers. It does
        # not merge, disable, unload or duplicate their weights.
        backbone = model.get_base_model() if callable(getattr(model, 'get_base_model', None)) else model
        backbone.eval()
        self.engine = Encbank(backbone, resume_j=resume_j, top_prepay_b=0,
                            block_diagonal=False, tokenizer=tokenizer)
        self.bits, self.group_size = bits, group_size
        self.reader_binding = str(reader_binding)
        self.device = self.engine.device
        if self.engine.dtype != torch.float16:
            raise ValueError('The qualified reader contract requires native FP16')
        if any(p.device != self.device for p in backbone.parameters()):
            raise ValueError('All model/adapter parameters must remain on one device')
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Only explicit CPU qualification or native CUDA is supported')
        config = self.engine.config
        self.bos_token_id = self._resolve_id(bos_token_id, tokenizer, config, 'bos_token_id')
        self.eos_token_id = self._resolve_id(eos_token_id, tokenizer, config, 'eos_token_id')
        if self.bos_token_id is None:
            raise ValueError('A frozen single BOS ID is required; never guess the first document token')

    @staticmethod
    def _resolve_id(explicit, tokenizer, config, name):
        value = explicit
        if value is None and tokenizer is not None:
            value = getattr(tokenizer, name, None)
        if value is None:
            value = getattr(config, name, None)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f'{name} must be one frozen nonnegative integer, not a silently narrowed list')
        return value

    def _check_entry(self, entry: HOnlyEntry):
        entry.assert_live()
        expected = (self.engine.resume_j, self.bits, self.group_size,
                    self.reader_binding, self.bos_token_id, self.eos_token_id,
                    self.device, self.engine.dtype)
        got = (entry.resume_j, entry.bits, entry.group_size,
               entry.reader_binding, entry.bos_token_id, entry.eos_token_id,
               entry.device, entry.dtype)
        if got != expected:
            raise ValueError('Entry depth/precision/reader/device/special-token binding mismatch')

    @torch.inference_mode()
    def encode_ids(self, context_ids, *, document_id: str = '',
                   progress: Callable[[dict], None] | None = None) -> HOnlyEntry:
        raw = _cpu_ids(context_ids, 'context_ids')
        if raw.numel() == 0:
            raise ValueError('A document must contain at least one token')
        packed_chunks = []
        self.last_reconstruction = []
        from codec_observation import reconstruction
        completed = 0
        for index, chunk in enumerate(raw.split(CHUNK_SIZE)):
            # One independent local-position chunk at a time. Packing is immediate;
            # no persistent all-native hidden list or document lower cache exists.
            hidden = self.engine.write_chunk(chunk)
            if hidden.dtype != torch.float16 or hidden.device != self.device:
                raise RuntimeError('Write changed the qualified hidden dtype/device')
            packed = quantize_tensor(hidden, bits=self.bits, group_size=self.group_size)
            self.last_reconstruction.append(reconstruction(hidden, packed, kind='chunk', index=index))
            packed_chunks.append(packed)
            del hidden, packed
            completed += chunk.numel()
            if progress is not None:
                progress({'completed_chunks': index+1,
                          'completed_document_tokens': completed,
                          'total_document_tokens': raw.numel()})
        sink_hidden = self.engine.write_chunk([self.bos_token_id])
        sink = quantize_tensor(sink_hidden, bits=16, group_size=self.group_size)
        self.last_reconstruction.append(reconstruction(sink_hidden, sink, kind='sink', index=None))
        del sink_hidden
        assert not self.engine._ctx_chunks and not self.engine._ctx_hj and self.engine._sink_hj is None
        return HOnlyEntry(str(document_id), raw, tuple(packed_chunks), sink,
                          self.engine.resume_j, self.bits, self.group_size,
                          self.reader_binding, self.bos_token_id, self.eos_token_id,
                          self.device, self.engine.dtype)

    def select(self, entry: HOnlyEntry, bare_question_ids) -> tuple[int, ...]:
        self._check_entry(entry)
        question = _cpu_ids(bare_question_ids, 'bare_question_ids')
        if question.numel() == 0:
            raise ValueError('A nonempty bare question is required for iterative BM25')
        indices = selectors.select_context_chunk_indices(
            SELECTOR, entry.raw_chunks(), question.tolist(), TOPK,
            needle_chunk_set=None, context_hj=None, query_hj=None,
            iter_rounds=ITER_ROUNDS, iter_hop_topk=ITER_HOP_TOPK)
        if (len(indices) > min(TOPK, len(entry.chunks)) or
            any(type(i) is not int or not 0 <= i < len(entry.chunks) for i in indices) or
            list(indices) != sorted(set(indices))):
            raise RuntimeError('Official iterative selector returned an invalid fixed-budget selection')
        return tuple(indices)

    @torch.inference_mode()
    def materialize_selected(self, entry: HOnlyEntry, indices: Sequence[int]) -> SelectedRead:
        self._check_entry(entry)
        indices = tuple(indices)
        if (len(indices) > TOPK or
            any(type(i) is not int or not 0 <= i < len(entry.chunks) for i in indices) or
            list(indices) != sorted(set(indices))):
            raise ValueError('Selection must be unique, in document order, and at most 12 chunks; an empty official selection is valid')
        hidden = tuple(entry.chunks[i].dequantize() for i in indices)
        sink_hidden = entry.sink.dequantize()
        entry_keys = {_storage_key(t) for _, t in entry.tensor_items()}
        for h in (*hidden, sink_hidden):
            if h.device != self.device or h.dtype != entry.dtype:
                raise RuntimeError('Read hidden dtype/device changed')
            if _storage_key(h) in entry_keys:
                raise RuntimeError('Read state aliases immutable entry backing')
        return SelectedRead(indices, hidden, sink_hidden)

    @torch.inference_mode()
    def generate_ids(self, entry: HOnlyEntry, query_ids, *, bm25_query_ids,
                     max_new_tokens: int = 48, capture_step_logits: bool = False) -> dict:
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('Official greedy decode requires a positive output cap')
        query = _cpu_ids(query_ids, 'query_ids')
        if query.numel() == 0:
            raise ValueError('Complete query token IDs cannot be empty')
        selected = self.select(entry, bm25_query_ids)
        request = self.materialize_selected(entry, selected)
        stats = {'capture_step_logits': bool(capture_step_logits)}
        try:
            # Exact official query-only lower prefill, contiguous causal upper
            # pack, two-position cached decode, and first-token EOS suppression.
            generated = self.engine._decode_from_pack(
                request.sink_hidden, request.hidden, query.tolist(),
                self.eos_token_id, max_new_tokens, True, stats=stats,
                n_context_chunks=len(entry.chunks))
        finally:
            request.release()
        if any(t == self.eos_token_id for t in generated) and self.eos_token_id is not None:
            raise RuntimeError('Official returned IDs must exclude EOS')
        if not generated or len(generated) > max_new_tokens:
            raise RuntimeError('Unexpected official generation length')
        stopped_on_eos = len(generated) < max_new_tokens
        if stopped_on_eos and self.eos_token_id is None:
            raise RuntimeError('Short generation has no EOS termination contract')
        return {
            'generated_token_ids': generated,
            'stopped_on_eos': stopped_on_eos,
            'hit_generation_cap': len(generated) == max_new_tokens,
            'selected_chunk_indices': list(selected),
            'dequantized_chunk_indices': list(selected),
            'selector': SELECTOR, 'topk': TOPK,
            'iter_hop_topk': ITER_HOP_TOPK, 'iter_rounds': ITER_ROUNDS,
            'automatic_rounds': 3, 'chunk_size': CHUNK_SIZE,
            'read_len': stats['read_len'],
            'n_selected_chunks': stats['n_selected_chunks'],
            'n_context_chunks': stats['n_context_chunks'],
            'selected_read_released': request.released,
            'read_policy': 'official query-only lower + fresh causal upper + cached decode',
            'precision': f'H{self.bits}; native FP16 BOS sink',
            'stats': stats,
        }
