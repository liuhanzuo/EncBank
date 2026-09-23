"""Experimental vLLM-derived recurrent kernel accepting HF state layout directly.

Functional state updates preserve the input cache; no in-place ownership changes.
The optional normalization path emulates HF's intermediate dtype rounding.
This module is for isolated validation only, not production rollout.
"""
import os
import torch
import triton
from native_layout_kernel import hf_layout_recurrent_kernel


class NativeLayoutGdn:
    def __init__(self, fuse_hf_norm=False):
        self.fuse_hf_norm = fuse_hf_norm

    def __call__(self, query, key, value, g, beta, initial_state,
                 output_final_state, use_qk_l2norm_in_kernel=False, **kwargs):
        if os.environ.get('COMEM_ISOLATED_KERNEL_PILOT') != '1' or torch.is_grad_enabled():
            raise RuntimeError('Isolated inference only')
        if query.ndim != 4 or query.shape != key.shape or query.shape[1] != 1:
            raise ValueError('Only independent single-token decode rows')
        b, _, h, k = query.shape
        v = value.shape[-1]
        if value.shape[:3] != (b, 1, h) or g.shape != (b, 1, h) or beta.shape != (b, 1, h):
            raise ValueError('Invalid GDN input shapes')
        use_cache = kwargs.pop('use_cache', None)
        if use_cache is not None and not isinstance(use_cache, bool):
            raise ValueError('Invalid decoder metadata')
        if any(x is not None for x in kwargs.values()):
            raise ValueError('Unsupported kernel options')
        if initial_state is not None:
            if initial_state.shape != (b, h, k, v) or initial_state.dtype != torch.float32:
                raise ValueError('Expected FP32 HF recurrent state [B,H,K,V]')
            if not initial_state.is_contiguous():
                raise ValueError('This prototype requires contiguous HF state')
        if any(x.device != query.device for x in (key, value, g, beta)):
            raise ValueError('Device mismatch')
        if initial_state is not None and initial_state.device != query.device:
            raise ValueError('State device mismatch')
        if use_qk_l2norm_in_kernel and not self.fuse_hf_norm:
            query = query * torch.rsqrt((query*query).sum(-1, keepdim=True)+1e-6)
            key = key * torch.rsqrt((key*key).sum(-1, keepdim=True)+1e-6)
            use_qk_l2norm_in_kernel = False
        q, key, value, g, beta = [x.contiguous() for x in (query, key, value, g, beta)]
        output = torch.empty_like(value)
        final = torch.empty((b, h, k, v), device=query.device, dtype=torch.float32)
        bk, bv = triton.next_power_of_2(k), min(triton.next_power_of_2(v), 32)
        hf_layout_recurrent_kernel[(1, triton.cdiv(v, bv), b*h)](
            q=q, k=key, v=value, g=g, beta=beta, o=output,
            h0=initial_state, ht=final, cu_seqlens=None, ssm_state_indices=None,
            num_accepted_tokens=None, scale=1/(k**.5), N=b, T=1, B=b,
            H=h, HV=h, K=k, V=v, BK=bk, BV=bv,
            stride_init_state_token=h*k*v, stride_final_state_token=h*k*v,
            stride_indices_seq=1, stride_indices_tok=1, IS_BETA_HEADWISE=False,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            INPLACE_FINAL_STATE=False, IS_KDA=False, num_warps=1, num_stages=3)
        return output, final if output_final_state else None
