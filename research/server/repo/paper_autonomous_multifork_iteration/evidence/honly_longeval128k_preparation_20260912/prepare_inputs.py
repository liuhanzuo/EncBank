"""CPU-only original LongEval generator extraction and exact input projection."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import statistics
import string
import sys
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def token_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()

def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert os.environ.get('PYTHONHASHSEED') == '1234'
    plan = json.loads((HERE / 'preparation_plan.json').read_text('utf-8'))
    for rel, expected in plan['input_source_sha256'].items():
        assert sha(ROOT / rel) == expected, rel
    source_path = ROOT / plan['generator_source']
    source = source_path.read_text('utf-8')
    tree = ast.parse(source)
    function_names = {'_random_label', 'build_lines_prompt'}
    selected = [n for n in tree.body if
                isinstance(n, ast.FunctionDef) and n.name in function_names or
                isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id == '_PROMPT_HEADER' for t in n.targets)]
    assert len(selected) == 3
    env = {'random': random, 'string': string}
    exec(compile(ast.Module(body=copy.deepcopy(selected), type_ignores=[]),
                 str(source_path), 'exec'), env)
    build_node = next(n for n in selected if isinstance(n, ast.FunctionDef) and
                      n.name == 'build_lines_prompt')
    render = next(n for n in build_node.body if isinstance(n, ast.FunctionDef) and
                  n.name == 'render')
    query_assign = next(n for n in render.body if isinstance(n, ast.Assign) and
                        isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'query')
    query_fn = copy.deepcopy(render)
    query_fn.name = 'source_query_text'
    query_fn.body = [copy.deepcopy(query_assign), ast.Return(value=ast.Name(id='query', ctx=ast.Load()))]
    query_module = ast.fix_missing_locations(ast.Module(body=[query_fn], type_ignores=[]))
    exec(compile(query_module, str(source_path), 'exec'), env)
    # No import of upstream eval module, CoMem, model, scorer or oracle helper.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen3-8B',
                                               local_files_only=True, use_fast=True)
    assert tokenizer.is_fast
    assert tokenizer.eos_token_id == 151645
    assert tokenizer.convert_tokens_to_ids('<|endoftext|>') == 151643
    output = HERE / 'inputs'
    output.mkdir(exist_ok=False)
    (output / 'scoring_only').mkdir()
    items, labels, raw, lengths = [], [], [], []
    length_seed = 1234 + (zlib.crc32(b'128k') % 100000)
    header = env['_PROMPT_HEADER']
    for i in range(100):
        rng_seed = length_seed * 1000 + i
        prompt, expected, target_label, n_lines = env['build_lines_prompt'](
            131072, tokenizer, random.Random(rng_seed))
        source_query = env['source_query_text'](target_label)
        assert prompt.endswith(source_query)
        source_doc = prompt[:-len(source_query)]
        assert source_doc.startswith(header) and source_doc.endswith('\n')
        records = source_doc[len(header):].splitlines()
        assert len(records) == n_lines and n_lines % 64 == 0
        # Fixed separator ownership, identical for every query; no truncation.
        assert source_doc.endswith('>\n')
        document, query = source_doc[:-2], '>\n' + source_query
        assert document + query == prompt
        dids = tokenizer.encode(document, add_special_tokens=False)
        qids = tokenizer.encode(query, add_special_tokens=False)
        bare = f'line {target_label}'
        bids = tokenizer.encode(bare, add_special_tokens=False)
        plain_ids = tokenizer.encode(prompt, add_special_tokens=False)
        original_ids = tokenizer.encode(prompt, add_special_tokens=True)
        original_boundary_matches = (
            tokenizer.encode(source_doc, add_special_tokens=False) +
            tokenizer.encode(source_query, add_special_tokens=False) == plain_ids)
        assert dids + qids == plain_ids, f'BPE boundary mismatch at source index {i}'
        assert original_ids == plain_ids, 'Unexpected Qwen implicit special-token insertion'
        assert len(plain_ids) > 0  # Input-only: do not crop or reject values beyond the prior formal40960 guard.
        parsed = [re.fullmatch(r'line ([a-z]+-[a-z]+): REGISTER_CONTENT is <([0-9]{6})>', r)
                  for r in records]
        assert all(parsed)
        assert len({m.group(1) for m in parsed}) == n_lines
        # Generator integrity only; never passed to retrieval or used for selection.
        assert sum(m.group(1) == target_label and m.group(2) == expected for m in parsed) == 1
        item_id = f'longeval_128k_seed1234_{i:03d}'
        doc_id = 'longeval_doc_' + text_sha(document)
        row = {'id': item_id, 'document_id': doc_id, 'source_ordinal': i,
               'dataset': 'longeval_128k', 'document_token_ids': dids,
               'query_token_ids': qids, 'bare_question_token_ids': bids,
               'prefix_token_ids': [151643], 'max_new_tokens': 16, 'source_id': item_id, 'eos_token_id': 151645,
               'full_prompt_with_BOS_token_sha256': token_sha([151643] + plain_ids),
               'metadata': {'rng_seed': rng_seed, 'source_length_key': '128k',
                            'source_target_tokens': 131072, 'n_lines': n_lines,
                            'document_text_sha256': text_sha(document),
                            'query_text_sha256': text_sha(query),
                            'source_full_prompt_text_sha256': text_sha(prompt),
                            'source_full_prompt_token_sha256': token_sha(original_ids),
                            'separator_contract': 'fixed final record closing delimiter > and newline moved to full query prefix'}}
        items.append(row)
        labels.append({'id': item_id, 'document_id': doc_id, 'expected': expected})
        raw.append({'id': item_id, 'document_id': doc_id, 'document_text': document,
                    'query_text': query, 'bare_question_text': bare,
                    'source_document_final_separator': '>\n'})
        lengths.append({'id': item_id, 'source_ordinal': i, 'rng_seed': rng_seed,
                        'document_tokens': len(dids), 'query_tokens': len(qids),
                        'bare_question_tokens': len(bids), 'n_lines': n_lines,
                        'source_full_prompt_tokens': len(original_ids),
                        'full_prompt_with_BOS_tokens': len(plain_ids)+1,
                        'full_prompt_with_BOS_plus_generation_reserve': len(plain_ids)+17,
                        'document_chunks_512': (len(dids)+511)//512,
                        'source_naive_boundary_token_equal': original_boundary_matches,
                        'final_boundary_token_equal': True})
    assert len({r['document_id'] for r in items}) == 100
    assert len({r['id'] for r in items}) == 100
    save(output / 'inference_fixture.json', {'schema': 'longeval_complete_lines_128k100_input_v1',
         'items': items})
    save(output / 'scoring_only/labels.json', {'schema': 'longeval_numeric_exact_CPU_labels_v1',
         'items': labels, 'fixture_sha256': sha(output / 'inference_fixture.json')})
    with (output / 'raw_text.jsonl').open('x', encoding='utf-8') as stream:
        for row in raw:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    save(output / 'lengths.json', {'items': lengths})
    ranges = {}
    for key in ('document_tokens', 'query_tokens', 'bare_question_tokens',
                'source_full_prompt_tokens', 'full_prompt_with_BOS_tokens',
                'full_prompt_with_BOS_plus_generation_reserve', 'n_lines', 'document_chunks_512'):
        values = [r[key] for r in lengths]
        ranges[key] = {'min': min(values), 'max': max(values), 'mean': statistics.mean(values)}
    files = ['inference_fixture.json', 'scoring_only/labels.json', 'raw_text.jsonl', 'lengths.json']
    manifest = {'status': 'CPU_INPUTS_ONLY_HIGH_POSITION_MODEL_QUALIFICATION_PENDING',
        'items': 100, 'unique_documents': 100, 'cap': 16,
        'source_seed': 1234, 'length_seed': length_seed,
        'seed_formula': '(1234 + crc32(b"8k") % 100000) * 1000 + source_ordinal',
        'generator_source_sha256': sha(source_path),
        'generator_AST_sha256': hashlib.sha256(ast.dump(ast.Module(body=selected, type_ignores=[]),
                                        include_attributes=False).encode()).hexdigest(),
        'query_assignment_AST_sha256': hashlib.sha256(ast.dump(query_assign,
                                        include_attributes=False).encode()).hexdigest(),
        'preparation_source_sha256': sha(Path(__file__)),
        'plan_sha256': sha(HERE/'preparation_plan.json'),
        'python': sys.version, 'interpreter': sys.executable,
        'versions': {k: importlib.metadata.version(k) for k in ('transformers','tokenizers')},
        'tokenizer_class': type(tokenizer).__name__, 'tokenizer_is_fast': tokenizer.is_fast,
        'tokenizer_bos_metadata': tokenizer.bos_token_id, 'tokenizer_eos_metadata': tokenizer.eos_token_id,
        'explicit_prefix': [151643], 'proposed_scalar_eos': 151645,
        'ranges': ranges, 'all_source_prompts_and_final_token_boundaries_exact': True,
        'source_naive_boundary_equal_count': sum(r['source_naive_boundary_token_equal'] for r in lengths),
        'new_queries_or_context_crop': False, 'scores_computed': False,
        'retrieval_preselection_computed': False, 'model_loaded': False,
        'input_only_no_GPU_launchable_package': True,
        'over_prior_40960_guard': sum(x['full_prompt_with_BOS_plus_generation_reserve'] > 40960 for x in lengths),
        'over_nominal_target_including_BOS_query_cap': sum(x['full_prompt_with_BOS_plus_generation_reserve'] > 131072 for x in lengths),
        'tokenizer_model_or_ROPE_configs_modified': False,
        'high_position_numerics_qualified_here': False,
        'files': {p: {'sha256': sha(output/p), 'bytes': (output/p).stat().st_size} for p in files},
        'limits': ['One synthetic LongEval 128K cell, not five lengths or paper exact examples.',
                   'Original approximate generator may exceed nominal 131072; no crop.',
                   'Separate fixed BOS is explicit adapter policy relative to original Qwen plain CLI.',
                   'Formal method/resource/scorer integration and independent review remain pending.']}
    save(output/'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'items': 100, 'unique_documents': 100,
                      'ranges': ranges, 'manifest_sha256': sha(output/'manifest.json')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
