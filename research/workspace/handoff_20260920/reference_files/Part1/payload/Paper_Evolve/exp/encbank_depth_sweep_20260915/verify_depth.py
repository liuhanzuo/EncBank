"""Independently rescore outputs, then check depth-specific pairing and reuse."""
import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--index', type=int, required=True)
p.add_argument('--job', required=True)
a = p.parse_args()
plan = json.loads((ROOT / 'plan.json').read_text())
task = plan['jobs'][a.index]
run = ROOT / 'results' / task['name'] / f"j{task['j']:02}" / f'job_{a.job}'
protocol = json.loads((run / 'protocol.json').read_text())
anchor = Path(plan['models'][task['name']]['anchor'])
assert protocol['j'] == task['j'] and protocol['training']['steps'] == 200
assert not (run / 'failure.json').exists()
assert (run / 'adapter-final.pt').exists() and protocol['trainable_parameters'] > 0
for name in ('correctness.json', 'correctness_adapted.json'):
    checks = json.loads((run / name).read_text())
    assert checks['replay']['equal'] and checks['cache']['equal']
checks = json.loads((run / 'base_reuse_checks.json').read_text())
assert len(checks) == 3 and all(c['equal'] for c in checks)
with gzip.open(run / 'samples.jsonl.gz', 'rt', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f]
with gzip.open(anchor / 'samples.jsonl.gz', 'rt', encoding='utf-8') as f:
    original_samples = [json.loads(line) for line in f]
assert samples == original_samples, 'Depths must share every token and evidence selection'
base = {r['id']: r for r in map(json.loads, (anchor / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()) if r['arm'] == 'replay_base'}
records = [json.loads(line) for line in (run / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
for record in records:
    if record['arm'] == 'replay_base':
        assert record.pop('reused_from') == str(anchor / 'predictions.jsonl')
        assert record == base[record['id']]
training = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
assert [r['step'] for r in training] == list(range(1, 201))
subprocess.run([sys.executable, '-B', str(ROOT / 'pilot_support' / 'verify_results.py'), str(run)], check=True)
verified = json.loads((run / 'verified_summary.json').read_text())
verified.update(depth_pairing_verified=True, base_replay_reuse_verified=True, j=task['j'], L=protocol['L'],
                trainable_parameters=protocol['trainable_parameters'])
(run / 'verified_summary.json').write_text(json.dumps(verified, indent=2) + '\n')
(run / 'progress.json').write_text(json.dumps({'phase': 'complete', 'verified': True, 'j': task['j'], 'completed_samples': 120}, indent=2) + '\n')
