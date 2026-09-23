"""Durable four-arm remote SFT queue; never evict another GPU process."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/data/liuhanzuo/encbank_v2_20260908')
WORK = ROOT / 'workspace'
CODE = WORK / 'exp/beacon_encbank_20260909'
OUT = ROOT / 'outputs/beacon_sft_20260909'
ARMS = [('beacon4', 'beacon', 4), ('encbank', 'encbank', 1),
        ('beacon8', 'beacon', 8), ('pool4', 'pool', 4)]


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def free_gpus():
    xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=15)
    result = []
    for index, gpu in enumerate(ET.fromstring(xml).findall('gpu')):
        if '3090' not in gpu.findtext('product_name', ''):
            continue
        used = int(gpu.findtext('fb_memory_usage/used').split()[0])
        busy = []
        for process in gpu.findall('processes/process_info'):
            name = process.findtext('process_name', '')
            if 'Xorg' not in name and 'gnome-shell' not in name:
                busy.append(name)
        if used < 512 and not busy:
            result.append(index)
    return result


def command(arm, mode, ratio, phase):
    destination = OUT / arm
    args = [sys.executable, '-u', str(CODE / 'train_sft.py'),
            '--model', str(ROOT / 'models/Qwen3-8B'),
            '--init-adapter', str(ROOT / 'outputs/8b_j12_pub_4k/final/adapter.pt'),
            '--train', str(CODE / 'data/qasper_sft/train.jsonl'), '--dev', str(CODE / 'data/qasper_sft/dev.jsonl'),
            '--out', str(destination), '--device', 'cuda:0', '--mode', mode,
            '--ratio', str(ratio), '--steps', '500', '--grad-accum', '4',
            '--eval-limit', '32', '--final-eval-limit', '100', '--max-new-tokens', '128']
    if phase == 'smoke':
        args += ['--stop-after', '2']
    if (destination / 'last.pt').exists():
        args += ['--resume', str(destination / 'last.pt')]
    return args


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'controller.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A controller already owns this queue')
    # An interrupted controller cannot infer ownership of surviving trainers.
    # The launcher only restarts it once their PIDs are confirmed gone.
    state_path = OUT / 'queue.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    for item in previous.get('jobs', {}).values():
        if item.get('phase') == 'running':
            try:
                os.kill(item['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('A previous trainer still runs; do not launch a duplicate')
    jobs, running = {}, {}
    for arm, mode, ratio in ARMS:
        saved = OUT / arm / 'status.json'
        status = json.loads(saved.read_text()) if saved.exists() else {}
        jobs[arm] = {'mode': mode, 'ratio': ratio,
                     'phase': 'complete' if status.get('complete') else 'queued',
                     'next': 'train' if status.get('step', 0) >= 2 else 'smoke'}

    def persist():
        write_json(state_path, {'pid': os.getpid(), 'host': os.uname().nodename,
            'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'jobs': jobs, 'complete': all(x['phase'] == 'complete' for x in jobs.values()),
            'scope': 'single-GPU independent SFT arms; accuracy only; no inference speed claims'})

    while True:
        for arm, (proc, log, gpu_lock) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
            gpu_lock.close()
            item = jobs[arm]
            status_path = OUT / arm / 'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            item['returncode'] = rc
            item['step'] = status.get('step')
            if rc != 0:
                item['phase'] = 'failed'
            elif item['next'] == 'smoke' and status.get('step') == 2 and status.get('gradient_check'):
                item['phase'], item['next'] = 'queued', 'train'
            elif status.get('complete') and status.get('step') == 500:
                item['phase'] = 'complete'
            else:
                item['phase'], item['error'] = 'failed', 'Unexpected trainer status'
            del running[arm]
            persist()
        if all(x['phase'] in {'complete', 'failed'} for x in jobs.values()):
            persist()
            return
        reserved = {jobs[arm]['gpu'] for arm in running}
        candidates = [gpu for gpu in free_gpus() if gpu not in reserved]
        # Prefer the three presently empty cards; GPU0 is used only after its
        # existing G-type Python tasks exit. XML includes graphics processes.
        candidates.sort(key=lambda gpu: (gpu == 0, gpu))
        for arm, item in jobs.items():
            if item['phase'] != 'queued' or not candidates:
                continue
            gpu = candidates.pop(0)
            gpu_lock = (ROOT / 'logs' / f'beacon_sft_gpu{gpu}.lock').open('a')
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                gpu_lock.close()
                continue
            if gpu not in free_gpus():
                gpu_lock.close()
                continue
            destination = OUT / arm
            destination.mkdir(exist_ok=True)
            logfile = destination / (item['next'] + '.log')
            log = logfile.open('a', encoding='utf-8')
            args = command(arm, item['mode'], item['ratio'], item['next'])
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
                   'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
                   'HF_HUB_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
            proc = subprocess.Popen(args, cwd=WORK, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            item.update(phase='running', gpu=gpu, pid=proc.pid, command=args, log=str(logfile))
            running[arm] = (proc, log, gpu_lock)
            persist()
        persist()
        time.sleep(15)


if __name__ == '__main__':
    main()
