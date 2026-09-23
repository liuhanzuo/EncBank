from __future__ import annotations
import math
from dataclasses import dataclass
import torch
SUPPORTED_BITS = (2, 4, 8, 16)


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _pack_unsigned(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return values.to(torch.uint8)
    values_per_byte = 8 // bits
    if values.shape[-1] % values_per_byte:
        raise ValueError("the packed dimension must fill complete bytes")
    grouped = values.to(torch.int16).reshape(
        *values.shape[:-1], values.shape[-1] // values_per_byte, values_per_byte
    )
    shifts = (
        torch.arange(values_per_byte, device=values.device, dtype=torch.int16)
        * bits
    )
    return torch.sum(grouped << shifts, dim=-1).to(torch.uint8)


def _unpack_unsigned(
    packed: torch.Tensor, bits: int, output_length: int
) -> torch.Tensor:
    if bits == 8:
        return packed[..., :output_length]
    values_per_byte = 8 // bits
    shifts = (
        torch.arange(values_per_byte, device=packed.device, dtype=torch.int16)
        * bits
    )
    mask = (1 << bits) - 1
    unpacked = (
        (packed.to(torch.int16).unsqueeze(-1) >> shifts) & mask
    ).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :output_length].to(torch.uint8)


@dataclass
class PackedTensor:
    """Affine group quantization for an arbitrary floating-point cache tensor."""

    bits: int
    group_size: int
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    data: torch.Tensor
    scales: torch.Tensor | None = None
    biases: torch.Tensor | None = None
    squared_error_sum: float = 0.0
    reference_squared_sum: float = 0.0
    max_abs_error: float = 0.0

    @property
    def nbytes(self) -> int:
        return sum(
            tensor_nbytes(tensor)
            for tensor in (self.data, self.scales, self.biases)
        )

    def dequantize(self) -> torch.Tensor:
        elements = math.prod(self.original_shape)
        if self.bits == 16:
            # ``reshape().to(same_dtype)`` may return a view of the persistent
            # packed store.  Cache leaves are mutable (notably Qwen3.5
            # recurrent/conv state), so every fork must own independent
            # storage even though Q16 does not require numeric conversion.
            return (
                self.data.reshape(self.original_shape)
                .to(self.original_dtype)
                .clone()
            )
        assert self.scales is not None and self.biases is not None
        padded_elements = self.scales.numel() * self.group_size
        values = _unpack_unsigned(self.data, self.bits, padded_elements).float()
        grouped = values.reshape(-1, self.group_size)
        restored = grouped * self.scales.float().unsqueeze(-1)
        restored = restored + self.biases.float().unsqueeze(-1)
        return restored.reshape(-1)[:elements].reshape(self.original_shape).to(
            self.original_dtype
        )


def quantize_tensor(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int = 64,
) -> PackedTensor:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    if group_size < 1 or group_size % (8 // min(bits, 8)):
        raise ValueError("group size must fill complete packed bytes")
    if not tensor.is_floating_point():
        raise ValueError("only floating-point tensors can be quantized")
    original_shape = tuple(tensor.shape)
    flat = tensor.detach().reshape(-1)
    reference_squared_sum = float(torch.square(flat.float()).sum().item())
    if bits == 16:
        data = flat.clone()
        return PackedTensor(
            bits=bits,
            group_size=group_size,
            original_shape=original_shape,
            original_dtype=tensor.dtype,
            data=data,
            reference_squared_sum=reference_squared_sum,
        )
    if flat.numel() == 0:
        empty = torch.empty(0, device=tensor.device, dtype=torch.uint8)
        metadata = torch.empty(0, device=tensor.device, dtype=torch.bfloat16)
        return PackedTensor(
            bits=bits,
            group_size=group_size,
            original_shape=original_shape,
            original_dtype=tensor.dtype,
            data=empty,
            scales=metadata,
            biases=metadata,
        )

    groups = (flat.numel() + group_size - 1) // group_size
    padded_elements = groups * group_size
    if padded_elements != flat.numel():
        padding = flat[-1:].expand(padded_elements - flat.numel())
        flat = torch.cat([flat, padding])
    grouped = flat.float().reshape(groups, group_size)
    biases = grouped.amin(dim=-1)
    maxima = grouped.amax(dim=-1)
    levels = (1 << bits) - 1
    scales = (maxima - biases) / levels
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = torch.round((grouped - biases.unsqueeze(-1)) / scales.unsqueeze(-1))
    quantized = quantized.clamp_(0, levels).to(torch.uint8)
    restored = quantized.float() * scales.unsqueeze(-1) + biases.unsqueeze(-1)
    error = grouped.reshape(-1)[: tensor.numel()] - restored.reshape(-1)[: tensor.numel()]
    packed = _pack_unsigned(quantized.reshape(-1), bits)
    return PackedTensor(
        bits=bits,
        group_size=group_size,
        original_shape=original_shape,
        original_dtype=tensor.dtype,
        data=packed,
        scales=scales.to(torch.bfloat16),
        biases=biases.to(torch.bfloat16),
        squared_error_sum=float(torch.square(error).sum().item()),
        reference_squared_sum=reference_squared_sum,
        max_abs_error=float(error.abs().max().item()),
    )

