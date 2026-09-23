"""Remote-only, bounded official-data resource probes and matched SFT pilot.

Waits for the existing Qasper campaign; shares its per-GPU locks and never evicts
other processes. All four stress probes must pass before the 1,000-conversation
pilot is allowed to start. Failed jobs stop promotion instead of retrying forever.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/data/liuhanzuo/comem_v2_20260908')
CODE = ROOT / 'workspace/exp/beacon_comem_20260909'
ARMS = ('beacon4', 'comem', 'beacon8', 'pool4')


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def publish(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temp.replace(path)


def available_gpus(xml):
    free = []
    for index, gpu in enumerate(ET.fromstring(xml).findall('gpu')):
        if '3090' not in gpu.findtext('product_name', ''):
            continue
        used = int(gpu.findtext('fb_memory_usage/used').split()[0])
        busy = [p.findtext('process_name', '') for p in gpu.findall('processes/process_info')
                if not any(n in p.findtext('process_name', '') for n in ('Xorg', 'gnome-shell'))]
        if used < 512 and not busy:
            free.append(index)
    return sorted(free, key=lambda value: (value == 0, value))


def free_gpus():
    return available_gpus(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=15))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def file_identity(path):
    value = Path(path).stat()
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino


def qasper_complete(directory, dev_path=None):
    """Accept the original evaluator's actual schema, bound to its fixed dev100."""
    from aggregate_sft import load_evaluation, _same_inputs
    try:
        directory = Path(directory)
        dev_path = Path(dev_path) if dev_path is not None else CODE/'data/qasper_sft/dev.jsonl'
        if read_json(directory/'queue.json').get('complete') is not True:
            return False
        canonical = [json.loads(line) for line in dev_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if len(canonical) != 100 or len({row['id'] for row in canonical}) != 100:
            return False
        canonical.sort(key=lambda row: hashlib.sha256(f"42:{row['id']}".encode()).hexdigest())
        expected_ids = [row['id'] for row in canonical]
        expected_sha = digest(dev_path)
        reference = None
        for arm in ARMS:
            status = read_json(directory/arm/'status.json')
            recipe = read_json(directory/arm/'metadata.json')['recipe']
            if (status.get('complete') is not True or status.get('step') != 500
                    or status.get('target_steps') != 500 or recipe.get('steps') != 500
                    or recipe.get('seed') != 42 or recipe.get('dev_sha256') != expected_sha
                    or recipe.get('final_eval_limit') != 100 or recipe.get('dev_examples') != 100
                    or recipe.get('max_new_tokens') != 128):
                return False
            evaluation = load_evaluation(directory/arm/'eval_step500.json', 100)
            protocol = evaluation['protocol']
            if (evaluation['ids'] != expected_ids or protocol['decoding'] != 'greedy'
                    or protocol['max_new_tokens'] != 128 or protocol['enable_thinking'] is not False):
                return False
            eos = protocol['eos_token_ids']
            if not isinstance(eos, list) or not eos or any(type(i) is not int or i < 0 for i in eos):
                return False
            for row, source in zip(evaluation['records'], canonical):
                answers = source['answer']
                references = list(dict.fromkeys(([answers] if isinstance(answers, str) else answers)
                                               + (source.get('answers') or [])))
                if (any(row[key] != source[key] for key in ('document_id', 'source', 'split', 'question'))
                        or row['references'] != references or not isinstance(row.get('prediction'), str)):
                    return False
                ids = row.get('generated_ids')
                if (not isinstance(ids, list) or not ids or len(ids) > 128
                        or any(type(i) is not int or i < 0 for i in ids)
                        or row.get('generated_tokens') != len(ids) or any(i in eos for i in ids[:-1])):
                    return False
                if row.get('finish_reason') == 'eos':
                    if ids[-1] not in eos:
                        return False
                elif row.get('finish_reason') != 'max_new_tokens' or len(ids) != 128 or ids[-1] in eos:
                    return False
            if reference is not None and _same_inputs(reference, evaluation, expected_ids):
                return False
            reference = evaluation
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def trainer_options(pilot, arm, stage, *, steps, grad_accum, mlp_chunk_tokens=0):
    """Shared CLI/expected-recipe settings; no implicit trainer hyperparameters."""
    pilot = Path(pilot)
    prefix = 'probe_' if stage == 'probe' else ''
    options = dict(model=str(ROOT/'models/Qwen3-8B'),
        init_adapter=str(ROOT/'outputs/8b_j12_pub_4k/final/adapter.pt'),
        train_prepared=str(pilot/f'{prefix}train.jsonl'), dev_prepared=str(pilot/f'{prefix}dev.jsonl'),
        selection_json=str(pilot/('probe_selection.json' if stage == 'probe' else 'main_selection.json')),
        arm=arm, sample_order='selection_order' if stage == 'probe' else 'seeded_shuffle',
        steps=steps, grad_accum=grad_accum, j=12, rank=32, alpha=32.0, lr=2e-5,
        beacon_lr=1e-3, warmup=20, seed=42, head_chunk_size=128, save_every=10,
        eval_every=50, eval_ce_limit=25, eval_per_source=5,
        qa_max_new_tokens=512, summary_max_new_tokens=2048)
    if mlp_chunk_tokens:
        options['mlp_chunk_tokens'] = mlp_chunk_tokens
    return options


def stage_expectations(pilot, stage, *, steps, grad_accum, receipt, mlp_chunk_tokens=0):
    """Validate current inputs once on CPU, before examining any old job output."""
    from train_official_sft import (PreparedDataset, validate_splits, validate_tokenizer_fingerprint,
        select_ids, sample_schedule, budget_at, evaluation_ids, object_digest)
    options = trainer_options(pilot, ARMS[0], stage, steps=steps, grad_accum=grad_accum,
                              mlp_chunk_tokens=mlp_chunk_tokens)
    prefix = 'probe_' if stage == 'probe' else ''
    paths = [Path(options[key]) for key in ('train_prepared', 'dev_prepared', 'selection_json', 'init_adapter')]
    paths.append(Path(pilot)/'selection.json')
    before = {str(path): file_identity(path) for path in paths}
    hashes = {key: digest(options[key]) for key in ('train_prepared', 'dev_prepared', 'selection_json', 'init_adapter')}
    for split in ('train', 'dev'):
        entry = receipt['files'][prefix+split]
        if (Path(options[split+'_prepared']) != Path(pilot)/entry['path']
                or hashes[split+'_prepared'] != entry['sha256']):
            raise ValueError('Current prepared data differs from completed pilot receipt')
    train, dev = PreparedDataset(options['train_prepared'], 'train'), PreparedDataset(options['dev_prepared'], 'dev')
    validate_splits(train, dev)
    tokenizer = validate_tokenizer_fingerprint(train, dev, options['model'])
    train_ids, dev_ids, flat = select_ids(train, dev, options['selection_json'])
    for split, dataset, ids in (('train', train, train_ids), ('dev', dev, dev_ids)):
        if (flat.get(split+'_ids') != receipt['selection'][prefix+split+'_ids']
                or set(ids) != set(dataset.ids) or len(ids) != receipt['files'][prefix+split]['conversations']):
            raise ValueError('Flat selection must preserve the complete current receipt ID order')
    schedule = sample_schedule(train_ids, steps*grad_accum, options['seed'], options['sample_order'])
    ce_ids, gen_ids = evaluation_ids(dev, dev_ids, options['seed'], options['eval_ce_limit'], options['eval_per_source'])
    recipe = {**options, **{key+'_sha256': value for key, value in hashes.items() if key != 'selection_json'},
        'selection_sha256': hashes['selection_json'], 'selected_train_ids': train_ids, 'selected_dev_ids': dev_ids,
        'schedule_sha256': object_digest(schedule), 'final_budget': budget_at(train, schedule, len(schedule)),
        'preparation_protocol': train.protocol, 'tokenizer_fingerprint': tokenizer['tokenizer_fingerprint'],
        'eval_ce_ids': ce_ids, 'eval_generation_ids': gen_ids,
        'objective': 'mean complete-assistant-token CE within each conversation, then equal mean across accumulated conversations',
        'writer': 'all original 512-token source chunks; no retrieval or target truncation',
        'mixed_pretraining_replay': False}
    if mlp_chunk_tokens:
        recipe['mlp_execution'] = 'whole-gated-mlp-token-block-checkpoint-v1'
    if any(file_identity(path) != identity for path, identity in before.items()):
        raise ValueError('Inputs changed during CPU validation')
    return {arm: {'recipe': {**recipe, 'arm': arm}, 'file_identities': before} for arm in ARMS}


def completed_job(directory, steps, *, expected):
    from train_official_sft import evaluation_signature, object_digest, validate_evaluation
    try:
        if any(file_identity(path) != identity for path, identity in expected['file_identities'].items()):
            return False
        directory, recipe = Path(directory), expected['recipe']
        status = read_json(directory/'status.json')
        gradient = status.get('gradient_check') or {}
        positive = lambda value: type(value) in (int, float) and math.isfinite(value) and value > 0
        if (recipe['steps'] != steps or read_json(directory/'metadata.json').get('recipe') != recipe
                or status.get('complete') is not True or status.get('step') != steps
                or status.get('target_steps') != steps or not status.get('model_state_id')
                or status.get('cursor') != steps*recipe['grad_accum'] or status.get('budget') != recipe['final_budget']
                or not positive(gradient.get('reader_norm')) or gradient.get('frozen_base_has_grad') is not False
                or (recipe['arm'].startswith('beacon') and not positive(gradient.get('beacon_norm')))):
            return False
        signature = evaluation_signature(steps, status['model_state_id'], object_digest(recipe),
            recipe['eval_ce_ids'], recipe['eval_generation_ids'], recipe['qa_max_new_tokens'], recipe['summary_max_new_tokens'])
        if not signature['ce_conversation_ids'] or not signature['generation_conversation_ids']:
            return False
        validate_evaluation(read_json(directory/f'eval_step{steps}.json'), signature)
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def trainer_command(pilot, destination, arm, stage, smoke, *, steps, grad_accum, mlp_chunk_tokens=0):
    destination = Path(destination)
    args = [sys.executable, '-u', str(CODE/'train_official_sft.py')]
    for key, value in trainer_options(pilot, arm, stage, steps=steps, grad_accum=grad_accum,
                                      mlp_chunk_tokens=mlp_chunk_tokens).items():
        args.extend(['--'+key.replace('_', '-'), str(value)])
    args.extend(['--out', str(destination), '--device', 'cuda:0'])
    if stage == 'probe':
        args.append('--skip-initial-eval')
    if smoke:
        args.extend(['--stop-after', '1' if stage == 'probe' else '2'])
    if (destination/'last.pt').exists():
        args.extend(['--resume', str(destination/'last.pt')])
    return args


def launch_trainer(command, *, gpu_lock, log, env):
    # flock belongs to the shared open-file description. The trainer must retain
    # its descriptor if this controller dies; only the GPU lock is inherited.
    return subprocess.Popen(command, cwd=ROOT/'workspace', env=env, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(gpu_lock.fileno(),))


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/beacon_official_sft_20260909')
    parser.add_argument('--allow-training', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--mlp-chunk-tokens', type=int, default=0)
    args = parser.parse_args()
    if args.mlp_chunk_tokens < 0:
        raise ValueError('MLP chunk size cannot be negative')
    if sys.platform == 'win32' or ROOT not in args.out.resolve().parents:
        raise ValueError('This controller is for the named remote experiment root only')
    args.out.mkdir(parents=True, exist_ok=True)
    singleton = (args.out/'controller.lock').open('a')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = read_json(args.out/'queue.json')
    for item in old.get('jobs', {}).values():
        if item.get('phase') == 'running':
            try:
                os.kill(item['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('A previous trainer PID still exists; inspect it before restarting')
    jobs, running = {}, {}
    for stage in ('probe', 'train'):
        for arm in ARMS:
            identity = stage+'/'+arm
            previous = old.get('jobs', {}).get(identity, {})
            jobs[identity] = {'stage': stage, 'arm': arm,
                'phase': 'failed' if previous.get('phase') == 'failed' and not args.retry_failed else 'queued'}

    def persist(reason):
        publish(args.out/'queue.json', {'pid': os.getpid(), 'host': os.uname().nodename,
            'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'reason': reason, 'jobs': jobs, 'pilot_dir': str(args.pilot_dir),
            'allow_training': args.allow_training,
            'mlp_chunk_tokens': args.mlp_chunk_tokens,
            'complete': all(j['phase'] == 'complete' for j in jobs.values()),
            'scope': '3090 training and development only; no inference timing claim'})

    selection, expectations = None, {}
    while True:
        if selection is None:
            found = read_json(args.pilot_dir/'selection.json')
            if found.get('complete') is not True:
                persist('waiting_for_complete_pilot_data')
                time.sleep(15); continue
            files = found['files']
            if files['train']['conversations'] != 1000 or files['probe_train']['conversations'] < 2:
                raise ValueError('Unexpected bounded pilot or probe budget')
            if not all((args.pilot_dir/files[n]['path']).is_file() for n in ('train', 'dev', 'probe_train', 'probe_dev')):
                raise ValueError('Pilot receipt is missing its data files')
            selection = found
            for stage in ('probe', 'train'):
                expectations[stage] = stage_expectations(args.pilot_dir, stage,
                    steps=files['probe_train']['conversations'] if stage == 'probe' else 250,
                    grad_accum=1 if stage == 'probe' else 4, receipt=found,
                    mlp_chunk_tokens=args.mlp_chunk_tokens)
            for identity, item in jobs.items():
                item['steps'] = files['probe_train']['conversations'] if item['stage'] == 'probe' else 250
                item['grad_accum'] = 1 if item['stage'] == 'probe' else 4
                destination = args.out/item['stage']/item['arm']
                item['out'] = str(destination)
                if completed_job(destination, item['steps'], expected=expectations[item['stage']][item['arm']]):
                    item['phase'] = 'complete'
        if not qasper_complete(ROOT/'outputs/beacon_sft_20260909'):
            persist('waiting_for_all_four_qasper_final_evaluations')
            time.sleep(15); continue
        for identity, (proc, log, gpu_lock) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close(); fcntl.flock(gpu_lock, fcntl.LOCK_UN); gpu_lock.close()
            item = jobs[identity]
            status = read_json(Path(item['out'])/'status.json')
            item.update(returncode=rc, last_saved_step=status.get('step'))
            if rc != 0:
                item['phase'] = 'failed'
            elif completed_job(item['out'], item['steps'], expected=expectations[item['stage']][item['arm']]):
                item['phase'] = 'complete'
            elif status.get('phase') == 'paused_smoke' and status.get('gradient_check'):
                item['phase'] = 'queued'
            else:
                item.update(phase='failed', error='Trainer exited without expected checkpoint and evaluation')
            del running[identity]
            persist('updated_finished_job')
        probes = [j for j in jobs.values() if j['stage'] == 'probe']
        probes_passed = all(j['phase'] == 'complete' for j in probes)
        if not running and any(j['phase'] == 'failed' for j in probes):
            persist('resource_probe_failed_no_promotion'); return
        if not running and probes_passed and not args.allow_training:
            persist('resource_probes_complete_training_not_enabled'); return
        if all(j['phase'] in {'complete', 'failed'} for j in jobs.values()):
            persist('complete' if all(j['phase'] == 'complete' for j in jobs.values()) else 'training_failed'); return
        reserved = {jobs[name]['gpu'] for name in running}
        candidates = [gpu for gpu in free_gpus() if gpu not in reserved]
        for identity, item in jobs.items():
            if item['phase'] != 'queued' or not candidates:
                continue
            if item['stage'] == 'train' and (not probes_passed or not args.allow_training):
                continue
            gpu = candidates.pop(0)
            # Share the original remote campaign's ownership namespace.
            gpu_lock = (ROOT/'logs'/f'beacon_sft_gpu{gpu}.lock').open('a')
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                gpu_lock.close(); continue
            if gpu not in free_gpus():
                gpu_lock.close(); continue
            destination = Path(item['out']); destination.mkdir(parents=True, exist_ok=True)
            status = read_json(destination/'status.json')
            smoke = status.get('step', 0) < (1 if item['stage'] == 'probe' else 2)
            command = trainer_command(args.pilot_dir, destination, item['arm'], item['stage'], smoke,
                                      steps=item['steps'], grad_accum=item['grad_accum'],
                                      mlp_chunk_tokens=args.mlp_chunk_tokens)
            log_path = destination/('smoke.log' if smoke else 'train.log')
            log = log_path.open('a', encoding='utf-8')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
                   'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1',
                   'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
            try:
                proc = launch_trainer(command, gpu_lock=gpu_lock, log=log, env=env)
            except OSError as error:
                log.close(); gpu_lock.close()
                item.update(phase='failed', error=f'Trainer launch failed: {error}')
                persist('trainer_launch_failed')
                continue
            item.update(phase='running', gpu=gpu, pid=proc.pid, command=command, log=str(log_path))
            running[identity] = proc, log, gpu_lock
            persist('running')
        persist('running' if running else 'waiting_for_free_3090')
        time.sleep(15)


if __name__ == '__main__':
    main()
