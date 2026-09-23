"""Scalar-only observations; native/packed/reconstructed tensors never leave device."""
import math
import torch

@torch.inference_mode()
def reconstruction(native, packed, *, kind, index):
    restored=packed.dequantize()
    assert native.device==restored.device==packed.data.device
    assert native.dtype==restored.dtype==torch.float16 and native.shape==restored.shape
    reference=native.float(); difference=restored.float()-reference
    finite=bool(torch.isfinite(reference).all().item() and torch.isfinite(difference).all().item())
    assert finite,'Nonfinite H reconstruction is an execution failure, not a zero error'
    squared_error_sum=float(difference.square().sum(dtype=torch.float64).item())
    reference_squared_sum=float(reference.square().sum(dtype=torch.float64).item())
    max_abs_error=float(difference.abs().max().item()) if native.numel() else 0.0
    return {'kind':kind,'chunk_index':index,'N':native.numel(),'shape':list(native.shape),
            'bits':packed.bits,'group_size':packed.group_size,'stored_tensor_bytes':packed.nbytes,
            'device':str(native.device),'native_dtype':str(native.dtype),'decoded_dtype':str(restored.dtype),
            'squared_error_sum':squared_error_sum,'reference_squared_sum':reference_squared_sum,
            'relative_L2':math.sqrt(squared_error_sum/reference_squared_sum) if reference_squared_sum else None,
            'zero_denominator':reference_squared_sum==0,'max_absolute_error':max_abs_error,
            'padding_excluded':True,'metadata_rounding_included':True,'tensor_host_export':False}

def aggregate(observations,kind):
    rows=[r for r in observations if r['kind']==kind]
    sse=sum(r['squared_error_sum'] for r in rows);reference=sum(r['reference_squared_sum'] for r in rows)
    return {'kind':kind,'N':sum(r['N'] for r in rows),'tensors':len(rows),
            'squared_error_sum':sse,'reference_squared_sum':reference,
            'relative_L2':math.sqrt(sse/reference) if reference else None,'zero_denominator':reference==0,
            'max_absolute_error':max((r['max_absolute_error'] for r in rows),default=0.0),
            'stored_tensor_bytes':sum(r['stored_tensor_bytes'] for r in rows)}
