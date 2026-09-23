"""CPU-only reconstruction of the completed Linux 16k RULER baseline draw."""
import argparse
import hashlib
import json
import locale
import os
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(ROOT / 'exp'), str(ROOT / 'Encbank')]
TASK_HASHES = {'niah_multikey_1': 7998353218535504284,
               'niah_single_2': -6132245302746163352,
               'variable_tracking': 4224176515762937225}


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--remote-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert sys.version_info[:2] == (3, 10) and sys.hash_info.algorithm == 'siphash24'
    assert os.environ.get('PYTHONHASHSEED') == '0'
    assert all(hash((task, '16k')) == value for task, value in TASK_HASHES.items())
    import torch
    from transformers import AutoTokenizer
    from eval import ruler as R
    from encbank import selectors
    torch.set_num_threads(2)
    model = args.remote_root / 'models/Qwen3-8B'
    essay = ROOT / 'exp/data/pg19_essay.txt'
    R._ESSAY_PATH = str(essay)
    R._ESSAY_WORDS_CACHE = None
    tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'status': 'preparing', 'purpose': 'matched controls for existing strong baselines',
                'python': sys.version, 'hash_info': str(sys.hash_info), 'pythonhashseed': '0',
                'preferred_encoding': locale.getpreferredencoding(False), 'seed': 42,
                'model': str(model), 'essay': str(essay), 'tasks': {}, 'source_sha256': {},
                'direct_baseline_fields': ['task', 'length', 'i', 'answers', 'n_tokens'],
                'pack_evidence': 'Reconstructed with the original token-based selector and saved recipe; old baseline rows did not persist selected pack indices.'}
    for path in [ROOT / 'exp/s15_ruler_lower.py', ROOT / 'Encbank/eval/ruler.py',
                 ROOT / 'Encbank/encbank/selectors.py', essay, model / 'tokenizer.json',
                 model / 'tokenizer_config.json']:
        manifest['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    baseline_paths = {'trained_pub': ('trained_pub', 'pub'), 'cacheblend16': ('cacheblend16', 'cacheblend16')}
    for task, task_hash in TASK_HASHES.items():
        baseline = {}
        for label, (folder, arm) in baseline_paths.items():
            path = args.remote_root / f'outputs/{folder}/full/ruler/{arm}/{task}_16k.json'
            obj = json.loads(path.read_text())
            assert len(obj['rows']) == 50 and {row['i'] for row in obj['rows']} == set(range(50))
            expected = {'j': 12, 'tasks': task, 'lengths': '16k', 'n': 50, 'topk': 12,
                        'selector': 'auto', 'iter_hop_topk': 4, 'max_new_tokens': 48, 'seed': 42}
            assert all(obj['args'][key] == value for key, value in expected.items()), (label, task)
            baseline[label] = {row['i']: row for row in obj['rows']}
        base_seed = 42 + task_hash % 100000
        icl = R._make_vt_icl(random.Random(base_seed + 777), 4) if task == 'variable_tracking' else None
        selector = 'iter_bm25' if task == 'variable_tracking' else 'bm25'
        rows = []
        for index in range(50):
            prompt, answers, gold = R._build_sample(task, R._LENGTH_TOKENS['16k'], tok,
                                                   random.Random(base_seed * 1000 + index), icl)
            input_ids = tok.encode(prompt, add_special_tokens=True)
            bare_question = tok.encode(R._bare_question(prompt), add_special_tokens=False)
            ids = torch.tensor(input_ids, dtype=torch.long)
            chunks = list(ids.split(512))
            contexts = chunks[:-1]
            selected = selectors.select_context_chunk_indices(selector, contexts, bare_question, 12,
                                                                iter_hop_topk=4, iter_rounds=0)
            direct = {'task': task, 'length': '16k', 'i': index,
                      'answers': answers, 'n_tokens': len(input_ids)}
            for label, expected_rows in baseline.items():
                assert all(expected_rows[index][key] == value for key, value in direct.items()), (label, task, index, direct)
            # Context/query boundaries follow the old trailing-512-token chunk interface.
            selected_ids = [int(value) for i in selected for value in contexts[i].tolist()]
            rows.append({**direct, 'prompt': prompt, 'gold': gold, 'input_ids': input_ids,
                         'bare_question_ids': bare_question, 'context_chunk_lengths': [len(c) for c in contexts],
                         'query_chunk_ids': chunks[-1].tolist(), 'selected_context_indices': selected,
                         'selector': selector, 'max_new_tokens': 60 if task == 'variable_tracking' else 48,
                         'input_ids_sha256': digest(input_ids), 'selected_context_ids_sha256': digest(selected_ids),
                         'query_chunk_sha256': digest(chunks[-1].tolist())})
        output = args.out / f'{task}_16k.jsonl'
        with output.open('w', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        manifest['tasks'][task] = {'file': output.name, 'n': len(rows), 'task_hash': task_hash,
                'base_seed': base_seed, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'baseline_direct_matches': {label: 50 for label in baseline},
                'context_pack_reconstructed': 50, 'selector': selector,
                'max_new_tokens': 60 if task == 'variable_tracking' else 48}
        print(json.dumps({'task': task, 'n': 50, 'baseline_matches': '50/50 for both baselines',
                          'base_seed': base_seed, 'fixture': str(output)}), flush=True)
    assert not torch.cuda.is_initialized(), 'Preparation must not initialize CUDA'
    manifest.update(status='ready', samples=150, existing_baseline_predictions_retained=300,
                    proposed_control_jobs=6, proposed_control_predictions=300, cuda_initialized=False)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status': 'ready', 'samples': 150, 'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    main()
