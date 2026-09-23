"""Persistent full-source packed H; original native hybrid query/decode protocol."""
from dataclasses import dataclass
import hashlib, struct
import torch
from transformers.cache_utils import DynamicCache
from extracted_minmax import quantize_tensor

def split_fixture(row):
    chunks = [row['input_ids'][i:i+512] for i in range(0, len(row['input_ids']), 512)]
    source, query = chunks[:-1], chunks[-1]
    selected = row['selected']
    assert selected == sorted(set(selected)) and len(selected) <= 12
    assert all(0 <= i < len(source) for i in selected)
    assert row['segments'] == [[row['sink']]] + [source[i] for i in selected] + [query]
    assert sum(map(len, source)) == row['source_tokens']
    return source, query, selected

@dataclass
class Entry:
    binding: str
    bits: int
    raw_ids: list
    chunks: list
    sink: torch.Tensor
    source_lengths: list
    released: bool = False

    def tensors(self):
        self.assert_live()
        yield self.sink
        yield from self.raw_ids
        for p in self.chunks:
            for t in (p.data, p.scales, p.biases):
                if t is not None: yield t

    def assert_live(self):
        assert not self.released, 'Released entry'

    def inventory(self):
        self.assert_live()
        hbytes = sum(p.nbytes for p in self.chunks)
        raw = sum(t.numel()*t.element_size() for t in self.raw_ids)
        sink = self.sink.numel()*self.sink.element_size()
        pointers = set(); allocated = 0
        for t in self.tensors():
            key = (str(t.device), t.untyped_storage().data_ptr())
            if key not in pointers:
                allocated += t.untyped_storage().nbytes(); pointers.add(key)
        assert allocated == hbytes+raw+sink
        return dict(bits=self.bits, blocks=len(self.chunks), source_tokens=sum(self.source_lengths),
                    packed_H_bytes=hbytes, sink_H16_bytes=sink, raw_ids_CPU_bytes=raw,
                    gpu_store_bytes=hbytes+sink, total_store_bytes=allocated,
                    document_lower_KV_bytes=0, group_size=64)

    def versions(self):
        return tuple((t.data_ptr(), t._version) for t in self.tensors())

    def digest(self):
        h = hashlib.sha256(self.binding.encode())
        for t in self.tensors():
            h.update(str((t.dtype, tuple(t.shape))).encode())
            h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return h.hexdigest()

    def release(self):
        self.raw_ids.clear(); self.chunks.clear(); self.source_lengths.clear()
        self.sink = None; self.released = True

@torch.no_grad()
def encode_entries(reader, source, sink, bits, binding):
    assert set(bits).issubset({4,8,16}) and bits
    entries = {}
    try:
        native_sink = reader.write([sink])
        for bit in bits:
            entries[bit] = Entry(binding, bit, [torch.tensor(c,dtype=torch.int64) for c in source],
                                 [], native_sink.clone(), list(map(len,source)))
        del native_sink
        for chunk in source:
            h = reader.write(chunk)
            assert h.dtype == torch.bfloat16 and h.device == reader.device
            for bit in bits: entries[bit].chunks.append(quantize_tensor(h, bits=bit, group_size=64))
            del h
        return entries
    except BaseException:
        for entry in entries.values(): entry.release()
        raise

def materialize(entry, selected, binding):
    entry.assert_live(); assert entry.binding == binding
    assert selected == sorted(set(selected))
    assert all(0 <= i < len(entry.chunks) for i in selected)
    return torch.cat([entry.sink.clone()] + [entry.chunks[i].dequantize() for i in selected], dim=1)

@torch.no_grad()
def generate(reader, entry, selected, query_ids, budget, eos, binding):
    assert budget > 0 and query_ids
    before = entry.versions()
    # These are request-local attention and DeltaNet caches, never entry fields.
    lower, upper = DynamicCache(config=reader.config), DynamicCache(config=reader.config)
    fixed = materialize(entry, selected, binding)
    query = reader.layers(reader.core.embed_tokens(reader.tensor(query_ids)), 0, reader.j, cache=lower)
    hidden = reader.layers(torch.cat([fixed,query],dim=1), reader.j, reader.L, cache=upper)
    lower_position = len(query_ids)
    upper_position = fixed.shape[1]+len(query_ids)
    del fixed, query
    generated = []
    for step in range(budget):
        token = int(reader.logits(hidden).argmax(-1).item()); generated.append(token)
        if token in eos or step == budget-1: break
        hidden = reader.core.embed_tokens(reader.tensor([token]))
        hidden = reader.layers(hidden,0,reader.j,cache=lower,offset=lower_position)
        hidden = reader.layers(hidden,reader.j,reader.L,cache=upper,offset=upper_position)
        lower_position += 1; upper_position += 1
    assert before == entry.versions(), 'Persistent entry mutated during Read'
    return generated
