"""Interleaved beacon residuals for Encbank: an experimental, trainable baseline.

This is NOT a reproduction of Activation Beacon: it uses a shared learnable
input embedding, ordinary frozen lower-layer attention, independently written
chunks, and Encbank's upper-layer reader. It stores no lower-layer document KV.
The supplied backbone must already have frozen lower layers and, optionally,
trainable reader LoRA. No pretrained model is loaded by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch import nn

ENCBANK_ROOT = Path(__file__).resolve().parents[2] / "Encbank"
if str(ENCBANK_ROOT) not in sys.path:
    sys.path.insert(0, str(ENCBANK_ROOT))
from encbank.model import Encbank


@dataclass
class BeaconMemory:
    hidden: torch.Tensor
    source_tokens: int
    compression_ratio: int
    split: int
    writer_id: str

    @property
    def tensor_bytes(self) -> int:
        return self.hidden.numel() * self.hidden.element_size()


@dataclass
class QueryState:
    logits: torch.Tensor
    bottom_cache: object
    top_cache: object
    query_position: int
    pack_position: int


class BeaconEncbank(nn.Module):
    """Keep only every inserted beacon's h_j; preserve Encbank query semantics.

    The last incomplete group also gets one beacon (ceil(T / ratio)), without
    adding padding tokens. Reading uses compact Encbank positions. The constructor
    does not attach LoRA or change the backbone's trainability.
    """

    FORMAT = "beacon-encbank-simplified-v1"

    def __init__(self, model: nn.Module, split: int, compression_ratio: int,
                 init_token_id: int, sink_token_id: int,
                 writer_id: str,
                 document_write_sink: bool = False,
                 gradient_checkpointing: bool = False):
        super().__init__()
        if isinstance(compression_ratio, bool) or not isinstance(compression_ratio, int) or compression_ratio < 1:
            raise ValueError("compression_ratio must be a positive integer")
        self.model = model
        if not isinstance(writer_id, str) or not writer_id.strip():
            raise ValueError("writer_id must identify the fixed backbone and writer checkpoint")
        self.writer_id = writer_id
        self.reader = Encbank(model, resume_j=split)
        if not 0 < split < self.reader.num_layers:
            raise ValueError("This content-bearing prototype requires 0 < split < L")
        for layer in self.reader.layers[:split]:
            if any(p.requires_grad for p in layer.parameters()):
                raise ValueError("Freeze the lower layers for this simplified writer")
        if any(p.requires_grad for p in self.reader.embed_tokens.parameters()):
            raise ValueError("Freeze ordinary token embeddings; beacon has its own parameter")
        vocab = self.reader.embed_tokens.weight.shape[0]
        if not 0 <= init_token_id < vocab or not 0 <= sink_token_id < vocab:
            raise ValueError("Initialization and sink token IDs must be in the vocabulary")
        self.ratio = compression_ratio
        self.init_token_id = int(init_token_id)
        self.sink_token_id = int(sink_token_id)
        self.document_write_sink = bool(document_write_sink)
        # A separate float32 master parameter; casts in forward retain gradients.
        initial = self.reader.embed_tokens.weight[init_token_id].detach().float().clone()
        self.beacon_embedding = nn.Parameter(initial)
        self.reader.grad_checkpoint = bool(gradient_checkpointing)

    def configuration(self) -> dict:
        return {"format": self.FORMAT, "writer_id": self.writer_id, "split": self.reader.resume_j,
                "compression_ratio": self.ratio, "hidden_size": self.reader.hidden_size,
                "init_token_id": self.init_token_id, "sink_token_id": self.sink_token_id,
                "document_write_sink": self.document_write_sink,
                "query_write_sink": False, "write_positions": "local-interleaved",
                "read_positions": "compact-encbank", "tail_policy": "append-one-no-padding"}

    def _sync_reader_location(self) -> None:
        # Encbank is an ordinary object: nn.Module.to() cannot update its snapshots.
        weight = self.reader.embed_tokens.weight
        self.reader.device, self.reader.dtype = weight.device, weight.dtype

    def interleave(self, token_ids) -> tuple[torch.Tensor, torch.Tensor]:
        """Build [x1..xr,b,x(r+1)..x(2r),b,...] in one causal sequence."""
        self._sync_reader_location()
        ids = self.reader._as_ids(token_ids)
        if ids.shape[1] == 0:
            raise ValueError("A document chunk must contain at least one token")
        raw = self.reader.embed_tokens(ids)
        beacon = self.beacon_embedding.to(device=raw.device, dtype=raw.dtype).view(1, 1, -1)
        pieces, beacon_rows = [], []
        offset = 0
        if self.document_write_sink:
            sink_ids = torch.tensor([[self.sink_token_id]], device=raw.device)
            pieces.append(self.reader.embed_tokens(sink_ids))
            offset = 1
        for start in range(0, ids.shape[1], self.ratio):
            segment = raw[:, start:start + self.ratio, :]
            pieces.extend((segment, beacon))
            offset += segment.shape[1]
            beacon_rows.append(offset)
            offset += 1
        rows = torch.tensor(beacon_rows, device=raw.device, dtype=torch.long)
        return torch.cat(pieces, dim=1), rows

    def write_chunk(self, token_ids) -> BeaconMemory:
        """Differentiable writer. Do not wrap in no_grad while training beacon."""
        hidden, rows = self.interleave(token_ids)
        source_tokens = hidden.shape[1] - rows.numel() - int(self.document_write_sink)
        pos = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        mask, rope = self.reader._make_mask_and_rope(hidden, pos)
        hidden = self.reader._run_layers(hidden, slice(0, self.reader.resume_j), mask, pos, rope)
        # A gather allocates only m rows; a strided slice could retain full storage.
        compact = torch.index_select(hidden, 1, rows)
        return BeaconMemory(compact, int(source_tokens), self.ratio, self.reader.resume_j, self.writer_id)

    @torch.no_grad()
    def encode_chunk(self, token_ids, cache_device="cpu") -> BeaconMemory:
        """Inference-only compact cache, without the document activation graph."""
        memory = self.write_chunk(token_ids)
        memory.hidden = memory.hidden.detach().to(cache_device).clone(memory_format=torch.contiguous_format)
        return memory

    def _validate_memory(self, memory: BeaconMemory) -> None:
        """Validate metadata without moving tensors or loading a GPU cache."""
        if memory.writer_id != self.writer_id:
            raise ValueError("Memory belongs to a different writer checkpoint")
        if memory.split != self.reader.resume_j or memory.compression_ratio != self.ratio:
            raise ValueError("Memory split/ratio differs from this reader")
        count = (memory.source_tokens + self.ratio - 1) // self.ratio
        if memory.source_tokens < 1 or tuple(memory.hidden.shape) != (1, count, self.reader.hidden_size):
            raise ValueError("Memory shape/source length is inconsistent")
        if not memory.hidden.is_floating_point():
            raise ValueError("Residual memory must have a floating point dtype")

    def _hidden_list(self, memories: Sequence[BeaconMemory]) -> list[torch.Tensor]:
        self._sync_reader_location()
        output = []
        for memory in memories:
            self._validate_memory(memory)
            output.append(memory.hidden.to(device=self.reader.device, dtype=self.reader.dtype))
        return output

    def read_logits(self, query_ids, memories: Sequence[BeaconMemory]) -> torch.Tensor:
        """Teacher-forced logits at every supplied query/answer-prefix position."""
        self._sync_reader_location()
        ids = self.reader._as_ids(query_ids)
        if ids.shape[1] == 0:
            raise ValueError("Query must contain at least one token")
        # Original Encbank query and sink, without beacon insertion or document KV.
        query = self.reader.write_chunk(ids)
        sink = self.reader.write_chunk([self.sink_token_id])
        return self.reader.read_core(sink, self._hidden_list(memories), query,
                                     logits_tail=ids.shape[1])

    def forward(self, document_chunks: Sequence, query_ids) -> torch.Tensor:
        memories = [self.write_chunk(chunk) for chunk in document_chunks]
        return self.read_logits(query_ids, memories)

    @torch.no_grad()
    def start_query(self, query_ids, memories: Sequence[BeaconMemory]) -> QueryState:
        self._sync_reader_location()
        ids = self.reader._as_ids(query_ids)
        if ids.shape[1] == 0:
            raise ValueError("Query must contain at least one token")
        query, bottom, local_pos = self.reader.write_prefill(ids)
        sink = self.reader.write_chunk([self.sink_token_id])
        logits, top, packed_pos = self.reader.read_prefill(sink, self._hidden_list(memories), query)
        return QueryState(logits, bottom, top, local_pos, packed_pos)

    @torch.no_grad()
    def decode_token(self, token_id: int, state: QueryState) -> torch.Tensor:
        self._sync_reader_location()
        logits = self.reader.decode_step(token_id, state.bottom_cache, state.top_cache,
                                          state.query_position, state.pack_position)
        state.query_position += 1
        state.pack_position += 1
        state.logits = logits
        return logits

    def save_trainable(self, path: str | Path) -> None:
        """Save beacon AND reader trainables; the frozen backbone is not copied."""
        tensors = {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}
        torch.save({"configuration": self.configuration(), "trainable": tensors}, path)

    def load_trainable(self, path: str | Path) -> None:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("configuration") != self.configuration():
            raise ValueError("Checkpoint configuration differs from this prototype")
        current = {name: p for name, p in self.named_parameters() if p.requires_grad}
        tensors = saved.get("trainable", {})
        if set(tensors) != set(current):
            raise ValueError("Checkpoint trainable names differ")
        for name, parameter in current.items():
            if tensors[name].shape != parameter.shape or not torch.isfinite(tensors[name]).all():
                raise ValueError(f"Invalid checkpoint tensor: {name}")
        with torch.no_grad():
            for name, parameter in current.items():
                parameter.copy_(tensors[name].to(parameter.device, parameter.dtype))

    def save_memory(self, memory: BeaconMemory, path: str | Path) -> None:
        self._validate_memory(memory)
        hidden = memory.hidden.detach().cpu().clone(memory_format=torch.contiguous_format)
        torch.save({"configuration": self.configuration(), "hidden": hidden,
                    "source_tokens": memory.source_tokens}, path)

    def load_memory(self, path: str | Path) -> BeaconMemory:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("configuration") != self.configuration():
            raise ValueError("Cache configuration differs from this prototype")
        memory = BeaconMemory(saved["hidden"], saved["source_tokens"], self.ratio,
                              self.reader.resume_j, self.writer_id)
        self._validate_memory(memory)
        return memory
