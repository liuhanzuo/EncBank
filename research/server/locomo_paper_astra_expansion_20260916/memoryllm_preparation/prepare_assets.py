"""CPU-only public asset download and isolated runtime preparation; no model load."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error
from email.utils import parsedate_to_datetime

HOME_ROOT = Path('/srv/encbank')
PROJECT = HOME_ROOT / 'qencbank_align_codex_20260911'
WORK = PROJECT / 'locomo_paper_astra_expansion_20260916/memoryllm_preparation'
MODEL = PROJECT / 'models/memoryllm-8b-chat-a8dec23c6ef9'
RUNTIME = HOME_ROOT / 'qencbank_runtime_20260911/memoryllm_20260916'
BASE_PYTHON = HOME_ROOT / 'qencbank_runtime_20260911/python312/bin/python'
REVISION = 'a8dec23c6ef973ec2253d81a20a0a76228801cef'


def own(path):
    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(HOME_ROOT) + '/'):
        raise ValueError(f'Outside own storage root: {resolved}')
    return resolved


def write_json(path, obj):
    path = own(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch_one(entry):
    name = entry['path']
    if '/' in name or name.startswith('.'):
        raise ValueError(name)
    target = own(MODEL / name)
    expected_sha = entry.get('lfs', {}).get('oid')
    size = entry['size']
    if target.exists() and target.stat().st_size == size:
        digest = sha256(target)
        if not expected_sha or digest == expected_sha:
            return {'path': str(target), 'size': size, 'sha256': digest, 'reused': True}
    partial = own(target.with_suffix(target.suffix + '.partial'))
    url = f'https://huggingface.co/YuWangX/memoryllm-8b-chat/resolve/{REVISION}/{name}'
    last_error = None
    for attempt in range(1, 6):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={start}-'} if start else {}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                mode = 'ab' if start and response.status == 206 else 'wb'
                with partial.open(mode) as stream:
                    for chunk in iter(lambda: response.read(8 << 20), b''):
                        stream.write(chunk)
            if partial.stat().st_size != size:
                raise ValueError(f'Expected {size} bytes, found {partial.stat().st_size}')
            digest = sha256(partial)
            if expected_sha and digest != expected_sha:
                raise ValueError(f'Hash mismatch for {name}')
            partial.replace(target)
            return {'path': str(target), 'size': size, 'sha256': digest, 'reused': False}
        except Exception as exc:
            last_error = repr(exc)
            delay = 2 * attempt
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                retry_after = exc.headers.get('Retry-After')
                try:
                    advertised = float(retry_after)
                except (TypeError, ValueError):
                    try:
                        advertised = parsedate_to_datetime(retry_after).timestamp() - time.time()
                    except Exception:
                        advertised = 0
                delay = max(300 * 2**(attempt - 1), advertised)
            write_json(WORK / 'download_status' / (name + '.json'),
                       {'status': 'RETRY' if attempt < 5 else 'FAILED', 'attempt': attempt,
                        'retry_delay_seconds': delay,
                        'error': last_error, 'bytes': partial.stat().st_size if partial.exists() else 0})
            if attempt < 5:
                time.sleep(delay)
    raise RuntimeError(f'{name}: {last_error}')


def download():
    own(MODEL).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((WORK / 'weight_manifest.json').read_text(encoding='utf-8-sig'))
    wanted = [e for e in manifest if e['path'].endswith(('.json', '.safetensors'))]
    start = time.time()
    write_json(WORK / 'download_status.json', {'status': 'RUNNING', 'pid': os.getpid(), 'files': len(wanted)})
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for result in pool.map(fetch_one, wanted):
            completed.append(result)
            write_json(WORK / 'download_status.json', {'status': 'RUNNING', 'pid': os.getpid(),
                       'completed': completed, 'files': len(wanted)})
    write_json(WORK / 'download_status.json', {'status': 'COMPLETE', 'pid': os.getpid(),
               'completed': completed, 'elapsed_seconds': time.time() - start, 'revision': REVISION})


def install():
    own(RUNTIME)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    for key, suffix in [('PIP_CACHE_DIR', 'pip_cache'), ('TMPDIR', 'tmp'), ('HF_HOME', 'hf_cache')]:
        p = own(WORK / suffix)
        p.mkdir(parents=True, exist_ok=True)
        env[key] = str(p)
    write_json(WORK / 'runtime_status.json', {'status': 'RUNNING', 'pid': os.getpid()})
    subprocess.run([str(BASE_PYTHON), '-m', 'venv', str(RUNTIME)], check=True, env=env)
    python = str(RUNTIME / 'bin/python')
    subprocess.run([python, '-m', 'pip', 'install', 'torch==2.5.1',
                    '--index-url', 'https://download.pytorch.org/whl/cu124'], check=True, env=env)
    subprocess.run([python, '-m', 'pip', 'install', 'transformers==4.48.2', 'peft==0.10.0',
                    'accelerate==1.2.0', 'numpy==1.26.4', 'einops==0.8.0', 'packaging==25.0',
                    'safetensors', 'sentencepiece'], check=True, env=env)
    freeze = subprocess.check_output([python, '-m', 'pip', 'freeze'], env=env, text=True)
    (WORK / 'runtime_freeze.txt').write_text(freeze)
    # Import official source only; do not construct a model or touch CUDA.
    env['PYTHONPATH'] = str(WORK / 'source')
    result = subprocess.check_output([python, '-c',
        'import json,torch,transformers,peft; from modeling_memoryllm import MemoryLLM; '
        'print(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__, '
        'peft=peft.__version__,cuda_initialized=torch.cuda.is_initialized(),class_name=MemoryLLM.__name__)))'],
        env=env, text=True)
    write_json(WORK / 'runtime_status.json', {'status': 'COMPLETE', 'pid': os.getpid(),
               'python': python, 'import_check': result, 'attention_backend': 'sdpa',
               'flash_attention_installed': False, 'model_constructed': False})


def start():
    own(WORK).mkdir(parents=True, exist_ok=True)
    for name in ['download', 'install']:
        status_path = WORK / (('download' if name == 'download' else 'runtime') + '_status.json')
        if status_path.exists():
            raise RuntimeError(f'Existing operation status requires inspection: {status_path}')
    owners = []
    for name in ['download', 'install']:
        log = own(WORK / f'{name}.log').open('ab', buffering=0)
        process = subprocess.Popen([str(BASE_PYTHON), str(own(__file__)), name],
                                   cwd=str(WORK), stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        owners.append({'operation': name, 'pid': process.pid, 'owner_uid': os.getuid(),
                       'command': [str(BASE_PYTHON), str(own(__file__)), name],
                       'started_at_epoch': time.time(), 'log': str(WORK / f'{name}.log')})
    write_json(WORK / 'operation_owners.json', {'operations': owners, 'work': str(WORK),
               'model': str(MODEL), 'runtime': str(RUNTIME), 'gpu_jobs': 0})
    print(json.dumps(owners))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'download', 'install'])
    parser.add_argument('--delay-seconds', type=float, default=0)
    args = parser.parse_args()
    action = args.action
    try:
        if args.delay_seconds:
            if action != 'download':
                raise ValueError('Delay is supported only for download resume')
            write_json(WORK / 'download_status.json', {'status': 'RATE_LIMIT_BACKOFF',
                       'pid': os.getpid(), 'resume_after_epoch': time.time() + args.delay_seconds})
            time.sleep(args.delay_seconds)
        {'start': start, 'download': download, 'install': install}[action]()
    except Exception as exc:
        if action != 'start':
            write_json(WORK / (('download' if action == 'download' else 'runtime') + '_status.json'),
                       {'status': 'FAILED', 'pid': os.getpid(), 'error': repr(exc)})
        raise
