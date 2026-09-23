"""Run the unchanged sparse training entry point with the Slurm lease backend.

The shim is confined to this new subprocess. It does not change any 3090 source
or relax a check: every existing validate_worker_lease call uses the Slurm guard.
"""
from __future__ import annotations
import sys
from types import ModuleType


def main():
    from slurm_gpu_guard import validate_worker_lease
    shim = ModuleType('remote_gpu_guard')
    shim.validate_worker_lease = validate_worker_lease
    previous = sys.modules.get('remote_gpu_guard')
    sys.modules['remote_gpu_guard'] = shim
    try:
        import train_sparse
        return train_sparse.main()
    finally:
        if previous is None:
            sys.modules.pop('remote_gpu_guard', None)
        else:
            sys.modules['remote_gpu_guard'] = previous


if __name__ == '__main__':
    main()
