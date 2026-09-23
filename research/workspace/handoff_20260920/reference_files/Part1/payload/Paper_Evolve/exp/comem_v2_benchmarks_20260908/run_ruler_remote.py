"""The existing RULER experiment with per-device admission on the remote queue."""
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'exp'))
sys.path.insert(0, str(ROOT / 'COMem'))


def admit_allocated_gpu(need_gb, cap_gb=None, **kwargs):
    import torch
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if os.environ.get('COMEM_REMOTE_QUEUE') != '1' or not visible.isdigit():
        raise RuntimeError('This wrapper requires one explicit GPU allocated by remote_queue.py')
    output = subprocess.check_output(['nvidia-smi', '-i', visible,
        '--query-gpu=memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    used, total, util = [int(x.strip()) for x in output.strip().split(',')]
    compute_pids = subprocess.check_output(['nvidia-smi', '-i', visible,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if used > 512 or util > 5 or compute_pids or total-used < need_gb*1024:
        raise RuntimeError(f'Allocated GPU {visible} is no longer idle: {output.strip()}, compute PIDs={compute_pids!r}')
    fraction = min(0.97, float(cap_gb or total/1024)*1024/total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f'[remote admission] physical GPU {visible}, {used}/{total} MiB used, cap={fraction:.3f}', flush=True)


if __name__ == '__main__':
    import gpu_gate
    import s15_ruler_lower
    original_build = s15_ruler_lower.build_arms

    def build_with_contextual_cacheblend(model, tokenizer, j, names, fix_layers):
        regular = [name for name in names if name != 'cacheblend16']
        arms = original_build(model, tokenizer, j, regular, fix_layers)
        if 'cacheblend16' in names:
            from comem import CoMem
            from cacheblend_contextual import ContextualCacheBlend
            arms['cacheblend16'] = ContextualCacheBlend(CoMem(model, j, tokenizer=tokenizer), .16)
            print('CacheBlend-style Qwen3 port: layer-1 V ranking, floor(.16*n_ctx), full two-layer bootstrap.', flush=True)
        return arms

    with patch.object(gpu_gate, 'acquire_gpu', admit_allocated_gpu), \
            patch.object(s15_ruler_lower, 'build_arms', build_with_contextual_cacheblend):
        s15_ruler_lower.main()
