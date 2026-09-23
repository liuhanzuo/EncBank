"""Use verified fixed 16k prompts and audit the selected pack in the existing reader."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(ROOT / 'exp'), str(ROOT / 'COMem')]


def take_flag(flag):
    index = sys.argv.index(flag)
    value = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    return value


def main():
    fixture = Path(take_flag('--frozen-inputs'))
    expected_sha = take_flag('--expected-fixture-sha256')
    validate_only = '--validate-only' in sys.argv
    if validate_only:
        sys.argv.remove('--validate-only')
    smoke_only = '--smoke-only' in sys.argv
    if smoke_only:
        sys.argv.remove('--smoke-only')
    manifest = json.loads((fixture.parent / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'ready' and manifest['samples'] == 150
    assert sys.version_info[:2] == (3, 10) and sys.hash_info.algorithm == 'siphash24'
    assert os.environ.get('PYTHONHASHSEED') == '0'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == expected_sha
    for path, digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, ('Source changed', path)
    rows = [json.loads(line) for line in fixture.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 50 and [row['i'] for row in rows] == list(range(50))
    count = int(sys.argv[sys.argv.index('--n') + 1])
    output = Path(sys.argv[sys.argv.index('--out') + 1])
    assert count == (1 if smoke_only else 50)
    assert not smoke_only or 'smoke' in output.parts
    rows = rows[:count]
    task = rows[0]['task']
    expected_args = {'--j': '12', '--tasks': task, '--lengths': '16k', '--n': str(count), '--topk': '12',
                     '--selector': 'auto', '--iter-hop-topk': '4', '--max-new-tokens': '48', '--seed': '42', '--check': '0'}
    for flag, value in expected_args.items():
        assert sys.argv[sys.argv.index(flag) + 1] == value, flag
    arm = sys.argv[sys.argv.index('--arms') + 1]
    assert arm in ('fix_all', 'j0') and '--adapter' not in sys.argv and '--adapter-pt' not in sys.argv
    assert hash((task, '16k')) == manifest['tasks'][task]['task_hash']
    import s15_ruler_lower as driver
    import torch
    from transformers import AutoTokenizer
    original_select = driver._sel.select_context_chunk_indices
    current = {'index': -1, 'selected_count': 0}

    def fixed_sample(received_task, target, tokenizer, rng, icl):
        current['index'] += 1
        row = rows[current['index']]
        assert received_task == task and target == driver.R._LENGTH_TOKENS['16k']
        assert tokenizer.encode(row['prompt'], add_special_tokens=True) == row['input_ids']
        assert tokenizer.encode(driver.R._bare_question(row['prompt']), add_special_tokens=False) == row['bare_question_ids']
        return row['prompt'], row['answers'], row['gold']

    def checked_select(selector, context_chunks, query_ids, topk, *args, **kwargs):
        row = rows[current['index']]
        assert selector == row['selector'] and topk == 12
        assert list(query_ids) == row['bare_question_ids']
        expected_chunks = [row['input_ids'][i:i + 512] for i in range(0, len(row['input_ids']), 512)][:-1]
        assert [chunk.tolist() for chunk in context_chunks] == expected_chunks
        selected = original_select(selector, context_chunks, query_ids, topk, *args, **kwargs)
        assert selected == row['selected_context_indices'], ('Pack changed', task, row['i'])
        current['selected_count'] += 1
        return selected

    if validate_only:
        tokenizer = AutoTokenizer.from_pretrained(manifest['model'], local_files_only=True)
        for row in rows:
            fixed_sample(task, driver.R._LENGTH_TOKENS['16k'], tokenizer, None, None)
            chunks = list(torch.tensor(row['input_ids']).split(512))
            checked_select(row['selector'], chunks[:-1], row['bare_question_ids'], 12, iter_hop_topk=4)
        assert not torch.cuda.is_initialized()
        print(json.dumps({'status': 'cpu_validated', 'task': task, 'arm': arm, 'inputs': count,
                          'packs': current['selected_count'], 'cuda_initialized': False}))
        return
    # Preserve incomplete attempts; publish the canonical result only after all checks.
    assert not output.exists(), 'Canonical output already exists; do not rerun a completed unit'
    attempt_dir = output.parent / (output.stem + '_attempts')
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (attempt_dir / f'{attempt:04d}.json').exists():
        attempt += 1
    raw_output = attempt_dir / f'{attempt:04d}.json'
    sys.argv[sys.argv.index('--out') + 1] = str(raw_output)
    with patch.object(driver.R, '_build_sample', fixed_sample), \
         patch.object(driver._sel, 'select_context_chunk_indices', checked_select):
        runpy.run_path(str(HERE / 'run_ruler_remote.py'), run_name='__main__')
    assert current['index'] == count - 1 and current['selected_count'] == count
    result = json.loads(raw_output.read_text())
    assert len(result['rows']) == count
    for actual, fixed in zip(result['rows'], rows):
        assert all(actual[key] == fixed[key] for key in ('task', 'length', 'i', 'answers', 'n_tokens'))
        for key in ('input_ids_sha256', 'selected_context_ids_sha256', 'query_chunk_sha256', 'selected_context_indices'):
            actual[key] = fixed[key]
    result['matched_input_fixture'] = {'path': str(fixture), 'sha256': expected_sha,
        'samples': count, 'selected_pack_checks': count, 'smoke_only': smoke_only,
        'raw_attempt_output': str(raw_output), 'preserved_baselines': ['trained_pub', 'cacheblend16']}
    result['args']['out'] = str(output)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=1))
    temporary.replace(output)


if __name__ == '__main__':
    main()
