"""Six-arm native Slurm LongEval bindings; check-only imports no model framework."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
ARMS = ('dense', 'h16', 'h8', 'h4', 'kivi2', 'kivi4')
PYTHON='/srv/encbank/qencbank_runtime_20260911/python312/bin/python'
REMOTE_ROOT='/srv/encbank/qencbank_align_codex_20260911/repo'
MODEL_ROOT='/srv/encbank/qencbank_align_codex_20260911/exact_local_reader'
RESOURCE={'allocator_cap_bytes':200*2**30,'minimum_free_bytes':220*2**30,'stable_idle_seconds':45}
def model_path(plan,part):
    assert part in ('model','adapter')
    p=Path(plan[part+'_root']); assert p==Path(MODEL_ROOT)/part
    p.resolve().relative_to(Path('/srv/encbank'))
    return p
CONFIG = {'resume_j': 12, 'chunk_size': 512, 'selector': 'iter_bm25', 'topk': 12,
          'iter_hop_topk': 4, 'iter_rounds': 0, 'H_group_size': 64,
          'KIVI_group_size': 32, 'KIVI_residual_length': 128,
          'bos_token_id': 151643, 'eos_token_id': 151645,
          'first_step_EOS_suppressed': True, 'max_new_tokens': 20, 'query_tokens_per_call': 1}

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'); tmp.replace(path)
def local(relative):
    path = (ROOT / relative).resolve(); path.relative_to(ROOT)
    assert not ({'.runtime', 'ko2', 'matched_v2', 'evaluator_workspaces', 'ground_truth'} & set(path.parts))
    return path
def token_sha(ids): return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
def document_groups(fixture):
    groups, documents = {}, {}
    for index, row in enumerate(fixture['items']):
        doc = row['document_id']
        if doc in documents: assert documents[doc] == row['document_token_ids']
        else: documents[doc] = row['document_token_ids']
        groups.setdefault(doc, []).append(index)
    return list(groups.items())
def load_function(path, module_name, function):
    """Exact source origin; no bare run_quality import after foreign sys.path additions."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, function)
    assert callable(fn) and Path(module.__file__).resolve() == path
    assert Path(fn.__code__.co_filename).resolve() == path
    return fn
def resolve_driver(plan):
    return (load_function(HERE/'run_quality.py', '_longeval_fp16_remote_run_quality', 'run'),
            load_function(local(plan['natural_qa_adapter']['path']), '_longeval_fp16_natural_qa', 'read_natural'))
def parser():
    assert __debug__, 'Python -O prohibited'
    p = argparse.ArgumentParser()
    for key in ('plan', 'expected-plan-sha256', 'arm', 'output'): p.add_argument('--'+key, required=True)
    p.add_argument('--check-only', action='store_true'); return p
def preflight(args,envelope=False):
    assert args.arm in ARMS and sha(args.plan) == args.expected_plan_sha256
    plan = read(args.plan)
    from backend_gate import validate_backend_binding
    validate_backend_binding(plan, allow_unbound=bool(getattr(args,'check_only',False)))
    assert plan['schema'] == 'BABILong_qa5_8k_100_FP16_six_method_remote_quality_v1'
    assert plan['arm_order'] == list(ARMS) and plan['configuration'] == CONFIG
    assert plan['backbone_dtype'] == 'float16' and plan['attention_backend'] == 'sdpa'
    assert not plan['training_or_runtime_offload'] and not plan['automatic_retry']
    for name, digest in plan['source_sha256'].items(): assert sha(local(name)) == digest, name
    # Labels/scorer are separate CPU analysis bindings. Model workers never open them.
    assert plan['resource_policy']==RESOURCE and plan['python']==PYTHON
    for key in ('fixture', 'activation', 'natural_qa_adapter'):
        spec = plan[key]; assert sha(local(spec['path'])) == spec['sha256'], key
    for name, spec in plan['qualification_bindings'].items(): assert sha(local(spec['path'])) == spec['sha256'], name
    for name, digest in plan['kivi_backend']['source_sha256'].items(): assert sha(local(plan['kivi_backend']['root'])/name) == digest, name
    fixture = read(local(plan['fixture']['path'])); items = fixture['items']; groups = document_groups(fixture)
    assert len(items) == len({r['id'] for r in items}) == plan['items_per_arm'] == 100
    assert len(groups) == plan['documents_per_arm'] == 100
    assert plan['expected_complete_phases_per_arm'] == 4*100 + 2*100 + 1 == 601
    for row in items:
        assert not ({'answer','answers','gold','reference','references','outputs','all_classes','evidence'} & set(row))
        assert row['dataset'] == 'babilong_qa5_8k' and row['max_new_tokens'] == 20
        assert row['prefix_token_ids'] == [151643] and row['eos_token_id'] == 151645
        assert row['document_token_ids'] and row['query_token_ids'] and row['bare_question_token_ids']
        assert all(type(t) is int and 0 <= t < 151936 for key in ('document_token_ids','query_token_ids','bare_question_token_ids') for t in row[key])
        full = [151643]+row['document_token_ids']+row['query_token_ids']
        assert token_sha(full) == row['full_prompt_with_BOS_token_sha256'] and len(full)+20 <= 40960
    activation = read(local(plan['activation']['path']))
    assert activation['status'] == 'completed_external_adapter_activated_for_inference_only'
    assert activation['external_status']['complete'] and activation['external_status']['step'] == 4000
    assert Path(args.output).resolve() == local(plan['outputs'][args.arm])
    out=Path(args.output)
    if envelope:
        rec=read(out/'execution.json')
        assert rec['plan_sha256']==args.expected_plan_sha256 and rec['arm']==args.arm
        assert not (out/'worker.json').exists() and not (out/'result.json').exists()
    else: assert not out.exists(), 'No duplicate output/resume'
    resolve_driver(plan)  # Safe stdlib-only path binding before any CUDA/model import.
    return plan, activation
