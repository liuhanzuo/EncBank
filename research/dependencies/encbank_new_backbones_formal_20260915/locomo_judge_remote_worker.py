"""Run fixed sanity checks and the complete saved LoCoMo cohort on gpu-node1."""
import concurrent.futures
import datetime
import fcntl
import json
import os
import time
import traceback

from astra_judge_batch import ROOT, run_batch, write_json
from codex_judge_linux import MODEL, call, version
from locomo_judge_reference import PROMPT

OUT = ROOT / 'judge_gpt6_astra' / 'formal_new_models'


def status(phase, **fields):
    write_json(OUT / 'worker_status.json', dict(phase=phase, pid=os.getpid(), at=time.time(),
               host='gpu-node1', user=os.environ.get('USER'), model=MODEL, **fields))


def classify(stimulus, out):
    prompt = PROMPT.format(**stimulus)
    for attempt in range(3):
        target = out if attempt == 0 else out / ('transport-retry-' + str(attempt))
        result = call(prompt, target)
        if result['ok']:
            return result
        messages = ' '.join(e.get('message', e.get('error', {}).get('message', ''))
                            for e in result.get('errors', [])).lower()
        forbidden = any(word in messages for word in ('403', '401', 'forbidden', 'unauthorized', 'quota'))
        transient = any(word in messages for word in (
            'overloaded', '502', '503', '504', 'timed out',
            'stream closed before response.completed', 'stream disconnected before completion'))
        if forbidden or not transient or attempt == 2:
            return result
        write_json(out / 'transport_retries.json', dict(retries=attempt+1,
                   reason='Transient upstream overload/network failure; no valid judgment received',
                   maximum_attempts=3, prompt_unchanged=True))
        time.sleep((10, 30)[attempt])
    raise AssertionError('Unreachable')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'worker.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert os.environ.get('MIDCACHE_JUDGE_API_KEY')
    assert not (OUT / 'STOP').exists()
    protocol = json.loads((ROOT / 'judge_protocol.json').read_text())
    assert protocol['model'] == MODEL and protocol['execution_host'] == 'gpu-node1'
    manifest = json.loads((ROOT / 'judge_inputs/manifest.json').read_text())
    assert manifest['verified_shards'] == 8
    inputs = []
    for entry in manifest['files']:
        path = ROOT / 'judge_inputs' / entry['file']
        assert path.parent == ROOT / 'judge_inputs'
        assert sum(1 for _ in path.open(encoding='utf-8')) == entry['records']
        inputs.append(path)
    assert sum(entry['records'] for entry in manifest['files']) == 27804
    client = version()
    status('CALIBRATING', client=client)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    calibration = ROOT / 'judge_gpt6_astra' / ('calibration-on-server-' + stamp)
    cases = json.loads((ROOT / 'judge_calibration_cases.json').read_text())
    def check(case):
        ident, question, gold, pred, expected = case
        result = call(PROMPT.format(question=question, gold=gold, pred=pred), calibration / ident)
        return dict(id=ident, expected=expected, returned=result['answer'].strip(),
                    transport_ok=result['ok'], client=result['client'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        checks = list(pool.map(check, cases))
    passed = len(checks) == 8 and all(row['transport_ok'] and row['returned'] == row['expected'] for row in checks)
    report = dict(passed=passed, model=MODEL, client=client, host='gpu-node1', n=len(checks), cases=checks,
                  scope='Fixed semantic/format sanity checks, not human-judge agreement')
    write_json(calibration / 'calibration.json', report)
    write_json(OUT / 'transport_calibration.json', report)
    if not passed:
        status('CALIBRATION_FAILED', calibration=str(calibration / 'calibration.json'))
        return
    protocol.update(client_version=client, calibration_passed=True, status='READY',
                    calibration_file=str(calibration / 'calibration.json'))
    write_json(ROOT / 'judge_protocol.json', protocol)
    write_json(OUT / 'protocol.json', protocol)
    status('JUDGING', client=client, available_records=27804, started_at=time.time())
    result = run_batch(inputs, OUT, protocol, workers=8, classifier=classify)
    if result['errors']:
        status('JUDGE_FAILURE', decisions=result['decisions'], errors=len(result['errors']))
        return
    assert result['available_complete'] and result['decisions'] == 27804
    assert all(cell['complete'] for arms in result['summary'].values() for cell in arms.values())
    status('COMPLETE', records=result['decisions'], oom=result['oom'], client=client)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        raise SystemExit('A remote Judge worker already holds the lock')
    except Exception:
        status('WORKER_ERROR', error=traceback.format_exc())
        raise
