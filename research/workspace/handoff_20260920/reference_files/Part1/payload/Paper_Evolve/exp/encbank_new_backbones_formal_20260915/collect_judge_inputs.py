"""Collect and validate all saved LoCoMo Judge inputs without calling a model."""
import collections
import datetime
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = '/srv/encbank/encbank_new_backbones_formal_20260915'


def main():
    out = ROOT / 'judge_gpt6_astra' / 'formal_new_models'
    inputs = out / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    command = f'/srv/encbank/Paper_Evolve/.venv/bin/python -B {REMOTE}/export_locomo_judge.py'
    response = subprocess.run(['ssh', '-o', 'ConnectTimeout=20', 'gpu-node1', command],
                              check=True, capture_output=True, text=True, timeout=60)
    manifest = json.loads(response.stdout)
    assert manifest['verified_shards'] == manifest['expected_shards'] == 8
    groups = collections.Counter()
    seen = set()
    questions = {}
    for item in manifest['files']:
        name = item['file']
        assert Path(name).name == name
        dest = inputs / name
        if not dest.exists():
            temp = dest.with_suffix('.download')
            subprocess.run(['scp', 'gpu-node1:' + REMOTE + '/judge_inputs/' + name, str(temp)],
                           check=True, capture_output=True, timeout=60)
            assert sum(1 for _ in temp.open(encoding='utf-8')) == item['records']
            temp.replace(dest)
        raw_path = ROOT / 'results_locomo' / item['cohort'] / f"shard{item['shard']}" / 'predictions.jsonl'
        raw = {}
        for line in raw_path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = (row['arm'], row['id'])
            assert key not in raw
            raw[key] = row
        rows = [json.loads(line) for line in dest.read_text(encoding='utf-8').splitlines()]
        assert len(rows) == len(raw) == item['records']
        for row in rows:
            assert row['cohort'] == item['cohort'] and row['shard'] == item['shard']
            key = (row['cohort'], row['arm'], row['id'])
            assert key not in seen
            seen.add(key)
            source = raw[(row['arm'], row['id'])]
            assert row['pred'] == source['text'] and row['answers'] == source['answers']
            assert row['category'] == source['extra']['category']
            assert row['status'] == source['status'] == 'ok'
            question = (row['question'], row['answers'], row['category'])
            assert questions.setdefault(row['id'], question) == question
            groups[(row['cohort'], row['arm'])] += 1
    assert len(seen) == 27804 and len(questions) == 1986
    assert len(groups) == 14 and set(groups.values()) == {1986}
    receipt = dict(at=datetime.datetime.now().astimezone().isoformat(),
                   generation_complete=True, verified_shards=8, available_records=len(seen),
                   normalized_inputs_match_raw=True, oom=0, judge_api_calls=0,
                   groups=[dict(cohort=m, arm=a, records=n) for (m, a), n in sorted(groups.items())])
    (out / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (out / 'generation_ready.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
