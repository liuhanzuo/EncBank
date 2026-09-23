"""Isolated, mathematically unchanged token-block execution of Qwen3's MLP.

Attention and the complete supervised sequence remain unchanged. This reduces
temporary MLP activations; it does not reduce persistent memory or cache size.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


EXECUTION_VERSION = "whole-gated-mlp-token-block-checkpoint-v1"


class ChunkedGatedMLP(nn.Module):
    """Compute gate/activation/up/product/down together for each token block.

    The ordinary Qwen3 MLP is token-independent, so this preserves its function
    and parameter gradients in exact arithmetic. GEMM shapes and gradient
    reduction order change, so floating-point results need not be bitwise equal.
    Non-reentrant checkpoints bound the intermediates retained during backward,
    including the original LoRA modules' FP32 casts. Full input/output tensors
    and attention/layer-checkpoint activations are still required.
    """

    def __init__(self, original, chunk_tokens):
        super().__init__()
        if type(chunk_tokens) is not int or chunk_tokens < 1:
            raise ValueError("chunk_tokens must be a positive integer")
        names = ("gate_proj", "up_proj", "down_proj", "act_fn")
        if any(not callable(getattr(original, name, None)) for name in names):
            raise ValueError("Expected Qwen3's ordinary gate/activation/up/down MLP")
        if set(original._modules) - set(names):
            raise ValueError("Unexpected extra MLP modules; do not silently change its function")
        # Re-register the very same objects under the very same child names.
        # In particular, original LoRALinear.forward and its A/B objects survive.
        for name in names:
            setattr(self, name, getattr(original, name))
        for name in ("config", "hidden_size", "intermediate_size"):
            if hasattr(original, name):
                setattr(self, name, getattr(original, name))
        self.chunk_tokens = chunk_tokens
        self.train(original.training)

    def _block(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x):
        # Keep real no-grad inference on the original full-sequence expression.
        # Training intentionally uses model.eval(), so .training is not a gate.
        if not torch.is_grad_enabled():
            return self._block(x)
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] == 0:
            return self._block(x)
        use_checkpoint = torch.is_grad_enabled() and (
            x.requires_grad or any(p.requires_grad for p in self.parameters()))
        pieces = []
        for start in range(0, flat.shape[0], self.chunk_tokens):
            block = flat[start:start+self.chunk_tokens]
            # Qwen3's ordinary MLP/LoRA contains no dropout or random operation.
            # PyTorch also preserves the caller's autocast state on recomputation.
            value = (checkpoint(self._block, block, use_reentrant=False,
                                preserve_rng_state=False) if use_checkpoint else self._block(block))
            pieces.append(value)
        result = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        return result.reshape(*x.shape[:-1], result.shape[-1])


def install_chunked_mlp(model, chunk_tokens):
    """Apply only to this dense Qwen3 model; preserve parameter/state-dict names.

    Every layer uses the same block size, including frozen lower layers. The
    lower writer otherwise runs unchanged, as do attention and the SFT loss.
    """
    if type(chunk_tokens) is not int or chunk_tokens < 0:
        raise ValueError("chunk_tokens must be a nonnegative integer")
    if chunk_tokens == 0:
        return []
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("MLP chunking is validated only for dense Qwen3")
    layers = list(model.model.layers)
    replacements = [ChunkedGatedMLP(layer.mlp, chunk_tokens) for layer in layers]
    for layer, replacement in zip(layers, replacements):
        layer.mlp = replacement
    return [f"model.layers.{i}.mlp" for i in range(len(layers))]
