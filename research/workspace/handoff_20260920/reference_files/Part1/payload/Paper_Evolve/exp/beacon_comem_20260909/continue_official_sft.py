"""Remote continuation: completed CPU pool -> validation -> bounded four-arm pilot."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/liuhanzuo/comem_v2_20260908')
CODE = ROOT/'workspace/exp/beacon_comem_20260909'
DATA = CODE/'data/activation_beacon_original'


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def ready(pool):
    return (read(pool/'pool_report.json').get('complete') is True
            and (pool/'eligible.jsonl').is_file()
            and (pool/'eligible_index.jsonl').is_file()
            and not (pool/'eligible.jsonl.part').exists())


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-pid', type=int, required=True)
    args = parser.parse_args()
    if sys.platform == 'win32' or not CODE.is_dir():
        raise ValueError('This continuation runs only in the named remote workspace')
    out = ROOT/'outputs/beacon_official_sft_20260909'
    pool, pilot = DATA/'prepared', DATA/'pilot1000_seed42'
    out.mkdir(parents=True, exist_ok=True)
    singleton = (out/'continuation.lock').open('a')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'host': os.uname().nodename,
             'preparation_pid': args.preparation_pid, 'pool': str(pool),
             'pilot': str(pilot), 'complete': False}

    def publish(phase, **extra):
        state.update(phase=phase, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
        temporary = out/'continuation.json.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
        temporary.replace(out/'continuation.json')

    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
           'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1'}

    def run(command, logfile):
        with (out/logfile).open('a', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT/'workspace', env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    try:
        while not ready(pool):
            publish('waiting_for_complete_cpu_pool', progress=read(pool/'progress.json'))
            try:
                os.kill(args.preparation_pid, 0)
            except ProcessLookupError:
                if not ready(pool):
                    raise RuntimeError('Preparation process exited before publishing a complete pool')
                break
            time.sleep(15)
        publish('validating_pool')
        run([sys.executable, str(CODE/'prepare_official_pool.py'), '--phase', 'validate',
             '--processed', str(DATA/'processed'), '--output', str(pool)], 'pool_validate.log')
        validation = read(pool/'pool_validation.json')
        if validation.get('passed') is not True:
            raise ValueError('Pool validation did not pass')
        publish('selecting_pilot', pool_validation=validation)
        if not read(pilot/'selection.json').get('complete'):
            staging = DATA/'pilot1000_seed42.preparing'
            if pilot.exists() or staging.exists():
                raise ValueError('Partial pilot directory exists; preserve and inspect before retrying')
            run([sys.executable, str(CODE/'select_official_pilot.py'),
                 '--pool', str(pool/'eligible.jsonl'), '--index', str(pool/'eligible_index.jsonl'),
                 '--out', str(staging), '--train-count', '1000', '--dev-per-source', '5',
                 '--seed', '42'], 'select_pilot.log')
            staging.rename(pilot)
        selection = read(pilot/'selection.json')
        from remote_official_sft_queue import ARMS, trainer_command
        publish('validating_all_training_inputs', files=selection['files'])
        for stage in ('probe', 'train'):
            steps = selection['files']['probe_train']['conversations'] if stage == 'probe' else 250
            for arm in ARMS:
                command = trainer_command(pilot, out/stage/arm, arm, stage, False,
                    steps=steps, grad_accum=1 if stage == 'probe' else 4)
                run(command+['--validate-only'], f'validate_{stage}_{arm}.log')
        publish('running_remote_queue')
        run([sys.executable, '-u', str(CODE/'remote_official_sft_queue.py'),
             '--pilot-dir', str(pilot), '--out', str(out), '--allow-training'], 'controller.log')
        queue = read(out/'queue.json')
        if queue.get('complete') is not True:
            raise RuntimeError('Remote queue stopped before all probes and pilot evaluations completed: '+str(queue.get('reason')))
        publish('complete', complete=True)
    except BaseException as exc:
        publish('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
