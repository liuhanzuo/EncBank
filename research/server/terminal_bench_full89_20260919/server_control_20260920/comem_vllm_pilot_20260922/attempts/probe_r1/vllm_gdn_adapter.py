"""Isolated Qwen GDN decode adapter; never imported by the running benchmark.

Targets the installed vLLM 0.22.1 fused recurrent operator. This is a decode
primitive, not a complete CoMem/vLLM model runner. HF state uses [B,H,K,V];
vLLM uses [B,H,V,K]. The owned transpose prevents mutation of the caller's state.
Keep this conversion for validation; a native-layout cache is a later change.
"""
import importlib.metadata
import os


class VllmGdnDecode:
    def __init__(self, kernel=None, preserve_hf_l2norm=True):
        self.kernel = kernel
        self.preserve_hf_l2norm = preserve_hf_l2norm

    def __call__(self, query, key, value, g, beta, initial_state,
                 output_final_state, use_qk_l2norm_in_kernel=False, **kwargs):
        import torch
        if torch.is_grad_enabled():
            raise RuntimeError('Inference only; use torch.inference_mode().')
        if query.ndim != 4 or query.shape[1] != 1 or query.shape != key.shape:
            raise ValueError('Only independent one-token decode rows are supported.')
        b, _, h, k = query.shape
        if value.ndim != 4 or value.shape[:3] != (b, 1, h):
            raise ValueError('HF already-expanded value/key head counts must agree.')
        v = value.shape[-1]
        if g.shape != (b, 1, h) or beta.shape != (b, 1, h):
            raise ValueError('Invalid gate shapes.')
        if kwargs.get('cu_seqlens') is not None or kwargs.get('cu_seq_lens_q') is not None:
            raise ValueError('Packed variable-length prefill requires a separate adapter.')
        extra = {k:v for k,v in kwargs.items() if v is not None}
        if extra:
            raise ValueError('Unsupported kernel options: '+', '.join(extra))
        if initial_state is None:
            state = torch.zeros((b,h,v,k),device=query.device,dtype=torch.float32)
        else:
            if initial_state.shape != (b,h,k,v) or initial_state.dtype != torch.float32:
                raise ValueError('Expected FP32 HF recurrent state [B,H,K,V].')
            if initial_state.device != query.device:
                raise ValueError('Recurrent state is on a different device.')
            state = initial_state.transpose(-1,-2).contiguous().clone()
        if any(x.device != query.device for x in (key,value,g,beta)):
            raise ValueError('All operator inputs must share a device.')
        # HF 5.16 normalizes before converting to FP32. Preserve that BF16
        # rounding initially; in-kernel FP32 normalization is a separate ablation.
        if use_qk_l2norm_in_kernel and self.preserve_hf_l2norm:
            query = query * torch.rsqrt((query * query).sum(-1,keepdim=True)+1e-6)
            key = key * torch.rsqrt((key * key).sum(-1,keepdim=True)+1e-6)
            use_qk_l2norm_in_kernel = False
        kernel = self.kernel
        if kernel is None:
            if importlib.metadata.version('vllm') != '0.22.1':
                raise RuntimeError('This prototype is pinned to installed vLLM 0.22.1.')
            if query.device.type != 'cuda' and os.environ.get('TRITON_INTERPRET') != '1':
                raise RuntimeError('Use CUDA, or explicitly enable the CPU Triton interpreter.')
            from vllm.model_executor.layers.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule
            kernel = self.kernel = fused_recurrent_gated_delta_rule
        output, final = kernel(q=query.contiguous(), k=key.contiguous(), v=value.contiguous(),
            g=g.contiguous(), beta=beta.contiguous(), initial_state=state,
            inplace_final_state=True, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        hf_final = final.transpose(-1,-2).contiguous() if output_final_state else None
        return output, hf_final


def install_isolated_decode_adapter(hf_qwen_module):
    """Explicit opt-in in a fresh pilot process only; no on-disk HF changes."""
    if os.environ.get('COMEM_ISOLATED_KERNEL_PILOT') != '1':
        raise RuntimeError('Refuse to patch without an explicit isolated pilot marker.')
    original = hf_qwen_module.torch_recurrent_gated_delta_rule
    replacement = VllmGdnDecode()
    hf_qwen_module.torch_recurrent_gated_delta_rule = replacement
    return original
