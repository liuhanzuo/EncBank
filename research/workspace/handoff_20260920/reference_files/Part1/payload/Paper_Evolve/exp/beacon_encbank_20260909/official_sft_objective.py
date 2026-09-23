"""Full-conversation SFT on Encbank memory, with bounded vocabulary activations.

This is an isolated experimental objective. It does not alter the running
Qasper trainer or launch a model. All document chunks are encoded once per
conversation; a causal reader pass supervises every assistant turn.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def conversation_loss(net, document_chunks, query_ids, labels, *, head_chunk_size=128):
    """Return mean CE over non-masked, already next-token-aligned labels.

    ``query_ids`` is the complete document-free chat except its last token.
    ``labels[t]`` is the following token when it belongs to an assistant answer,
    and -100 otherwise. The preparation module is responsible for that mask.
    Documents must contain only source text, never chat questions or targets.

    The connective, b=0 Encbank path is required. We preserve its causal attention
    and position remapping, selecting supervised query states before lm_head.
    Non-reentrant checkpointing of small CE blocks bounds retained vocabulary
    activations without truncating targets. This changes arithmetic scheduling,
    not the objective. CUDA memory feasibility still requires a real smoke test.
    """
    if head_chunk_size < 1:
        raise ValueError("head_chunk_size must be positive")
    net._sync_reader_location()
    reader = net.reader
    if reader.top_prepay_b != 0 or reader.block_diagonal:
        raise ValueError("Official SFT requires ordinary connective Encbank, b=0")
    ids = reader._as_ids(query_ids)
    targets = torch.as_tensor(labels, dtype=torch.long, device=reader.device)
    if targets.ndim != 1 or ids.shape[1] != targets.numel() or targets.numel() == 0:
        raise ValueError("Labels must align with every nonempty query position")
    supervised = targets.ne(-100).nonzero(as_tuple=True)[0]
    if supervised.numel() == 0:
        raise ValueError("Conversation has no assistant supervision")
    selected_targets = targets.index_select(0, supervised)
    if selected_targets.min() < 0 or selected_targets.max() >= reader.config.vocab_size:
        raise ValueError("Invalid target token ID")
    if not document_chunks or any(len(chunk) == 0 for chunk in document_chunks):
        raise ValueError("Need nonempty source document chunks")

    memories = [net.write_chunk(chunk) for chunk in document_chunks]
    query = reader.write_chunk(ids)
    sink = reader.write_chunk([net.sink_token_id])
    # Detached raw/pool writers still need upper-band activation checkpointing.
    # This leaf is not an optimized parameter and never enters a saved cache.
    if reader.grad_checkpoint and torch.is_grad_enabled() and net.mode != "beacon":
        sink = sink.detach().requires_grad_(True)
    packed = torch.cat([sink, *net._hidden_list(memories), query], dim=1)
    positions = torch.arange(packed.shape[1], device=reader.device).unsqueeze(0)
    mask, rope = reader._make_mask_and_rope(packed, positions)
    hidden = reader._run_layers(packed, slice(reader.resume_j, reader.num_layers),
                                mask, positions, rope)
    query_hidden = hidden[:, -ids.shape[1]:, :]
    selected = query_hidden[0].index_select(0, supervised)

    def block_ce(states, values):
        logits = reader.lm_head(reader.norm(states)).float()
        return F.cross_entropy(logits, values, reduction="sum")

    terms = []
    for start in range(0, supervised.numel(), head_chunk_size):
        states = selected[start:start + head_chunk_size]
        values = selected_targets[start:start + head_chunk_size]
        if torch.is_grad_enabled() and states.requires_grad:
            term = checkpoint(block_ce, states, values, use_reentrant=False)
        else:
            term = block_ce(states, values)
        terms.append(term)
    return torch.stack(terms).sum() / supervised.numel()
