"""Formal MemoryLLM wrapper using the existing native allocation/admission route."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace
from protocol import RESOURCE
from resource_guard import query, require_identity, stable_admission

HERE = Path(__file__).resolve().parent
HOME = Path('/srv/encbank')


def own(path):
    path = Path(path).resolve()
    path.relative_to(HOME)
    return path


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def preflight(plan, expected):
    own(HERE)
    assert sha(HERE / 'plan.json') == expected, 'Plan hash mismatch'
    for name, value in plan['source_sha256'].items():
        p = own(HERE / name)
        p.relative_to(HERE)
        assert sha(p) == value, name
    for group in ['asset_files', 'runtime_files']:
        for item in plan[group]:
            p = own(item['path'])
            assert p.is_file() and p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
    for key in ['output', 'model_root', 'runtime', 'task_cache_root']:
        # Existing runtime executable may symlink to system Python; writes never do.
        if key == 'runtime':
            own(Path(plan[key]).parent)
        else:
            own(plan[key])
    assert not Path(plan['output']).exists(), 'Output already exists; no blind resume'
    assert RESOURCE == plan['resource_policy']
    import torch, transformers, peft, accelerate
    actual = dict(torch=torch.__version__, transformers=transformers.__version__,
                  peft=peft.__version__, accelerate=accelerate.__version__)
    assert actual == plan['runtime_versions'], actual
    assert not torch.cuda.is_initialized(), 'CUDA initialized before admission'
    from run_locomo_memoryllm import load_inputs
    data = load_inputs(HERE / 'inference_inputs.json', plan['input_sha256'])
    return data, torch, actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan = read(HERE / 'plan.json')
    data, torch, versions = preflight(plan, args.expected_plan_sha256)
    if args.check_only:
        print(json.dumps({'status': 'PASS_FULL_ASSET_SOURCE_RUNTIME_INPUT_HASHES_NO_CUDA',
                          'assets': len(plan['asset_files']), 'runtime_files': len(plan['runtime_files']),
                          'source_ids': 1986, 'runtime_versions': versions, 'cuda_initialized': False}))
        return
    assert sys.prefix == plan['runtime_prefix']
    assert os.environ.get('SLURM_JOB_NAME') == plan['job_name']
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'backend:native'
    assert os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1'
    output = own(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'admitting', 'pid': os.getpid(), 'plan_sha256': args.expected_plan_sha256,
              'started_at_epoch': time.time(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'hostname': socket.gethostname(), 'resource_policy': RESOURCE,
              'quality_execution_started': False, 'scientific_result_complete': False}
    save(output / 'formal_execution.json', record)
    try:
        def capture(samples, elapsed):
            record['stable_interval_seconds'] = elapsed
            save(output / 'admission.json', {'samples': samples, 'stable_interval_seconds': elapsed})
        first = query()
        identity = require_identity(first, None)
        samples = stable_admission(identity, capture)
        assert identity['name'] == 'NVIDIA L20D'
        assert torch.cuda.device_count() == 1 and torch.cuda.get_allocator_backend() == 'native'
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        assert props.name == identity['name'] and [props.major, props.minor] == [10, 3]
        assert str(props.uuid).lower().removeprefix('gpu-') == identity['uuid'].lower().removeprefix('gpu-')
        assert torch.cuda.current_stream(0) == torch.cuda.default_stream(0)
        torch.cuda.set_per_process_memory_fraction(RESOURCE['allocator_cap_bytes'] / props.total_memory, 0)
        assert abs(torch.cuda.get_per_process_memory_fraction(0) * props.total_memory - RESOURCE['allocator_cap_bytes']) < 1
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)
        def forbid(*a, **kw):
            raise RuntimeError('Training/autograd is forbidden')
        torch.autograd.backward = forbid
        torch.autograd.grad = forbid
        record.update(status='running', physical_gpu_identity=identity, compute_capability=[10, 3],
                      qualified_allocator_cap_bytes=RESOURCE['allocator_cap_bytes'],
                      quality_execution_started=True)
        save(output / 'formal_execution.json', record)
        def gate():
            fresh = query()
            require_identity(fresh, identity)
            free, total = torch.cuda.mem_get_info(0)
            fresh.update(cuda_free_bytes=free, cuda_total_bytes=total)
            record.setdefault('fresh_pre_migration_checks', []).append(fresh)
            save(output / 'formal_execution.json', record)
            assert free >= RESOURCE['minimum_free_bytes'], 'Fresh native CUDA admission failed'
        from run_locomo_memoryllm import execute
        driver_args = SimpleNamespace(inputs=str(HERE / 'inference_inputs.json'),
            input_sha256=plan['input_sha256'], protocol_id=plan['protocol_id'],
            model_dir=plan['model_root'], model_revision=plan['model_revision'],
            source_dir=str(HERE / 'source'), output_dir=str(output), dtype='float16',
            seed=42, minimum_free_gib=220.0, resume=False)
        execute(driver_args, data, admission_gate=gate)
        result = read(output / 'execution.json')
        assert result['status'] == 'COMPLETE' and result['completed_predictions'] == 1986
        record.update(status='COMPLETE', scientific_result_complete=True,
                      prediction_sha256=sha(output / 'predictions.jsonl'))
    except BaseException as exc:
        record.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['finished_at_epoch'] = time.time()
        save(output / 'formal_execution.json', record)


if __name__ == '__main__':
    main()
