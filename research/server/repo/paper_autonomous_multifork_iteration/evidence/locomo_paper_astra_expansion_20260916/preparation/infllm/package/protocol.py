"""One-method RULER VT native Slurm bindings; check-only imports no model framework."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
ARMS = ('infllm_resident_qwen3',)
PYTHON='/srv/encbank/qcomem_runtime_20260911/python312/bin/python'
REMOTE_ROOT='/srv/encbank/qcomem_align_codex_20260911/repo'
MODEL_ROOT='/srv/encbank/qcomem_align_codex_20260911/exact_local_reader'
RESOURCE={'allocator_cap_bytes':200*2**30,'minimum_free_bytes':220*2**30,'stable_idle_seconds':45}
def model_path(plan,part):
    assert part in ('model','adapter')
    p=Path(plan[part+'_root']); assert p==Path(MODEL_ROOT)/part
    p.resolve().relative_to(Path('/srv/encbank'))
    return p
CONFIG = {'bos_token_id': 151643, 'eos_token_id': 151645, 'first_step_EOS_suppressed': True, 'max_new_tokens': 48, 'query_tokens_per_call': 1, 'document_prefill_chunk_size': 2048, 'n_init': 128, 'n_local': 4096, 'block_size': 128, 'max_cached_block': 32, 'topk': 16, 'exc_block_size': 512, 'repr_topk': 4, 'fattn': False, 'cache_strategy': 'lru', 'score_decay': None, 'chunk_topk_calc': None, 'async_global_stream': False, 'pin_memory': False, 'faiss': False, 'perhead': False}

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

N=1986
D=10
PHASES=4013
SCHEMA='LoCoMo1986_infllm_native_FP16_v1'
DATASET='locomo'
FAMILY='LoCoMo'

def load_fixture(plan):
    spec=plan['fixture'];assert sha(local(spec['path']))==spec['sha256']
    descriptor=read(local(spec['path']))
    assert descriptor['schema']=='LoCoMo1986_full_shared_documents_v1'
    files={}
    for key in ('documents','items'):
        binding=descriptor[key];path=local(binding['path']);assert sha(path)==binding['sha256']
        with path.open(encoding='utf-8') as stream:files[key]=[json.loads(line) for line in stream if line.strip()]
    docs={d['document_id']:d for d in files['documents']}
    assert len(docs)==len(files['documents'])==10 and len(files['items'])==1986
    assert set(docs)=={r['document_id'] for r in files['items']}
    forbidden={'answer','answers','gold','reference','references','evidence','category','source_qa','is_abstention'}
    for doc in docs.values():
        assert not forbidden & set(doc) and not doc['crop_applied']
        assert len(doc['document_token_ids'])==doc['document_tokens'] and token_sha(doc['document_token_ids'])==doc['document_token_sha256']
    for row in files['items']:
        assert not forbidden & set(row) and 'document_token_ids' not in row and 'source_id' not in row
        doc=docs[row['document_id']]
        assert row['source_conversation_index']==doc['source_conversation_index']
        row['document_token_ids']=doc['document_token_ids'];row['source_id']=row['id']
    return {'items':files['items'],'documents':files['documents'],'descriptor':descriptor}

def preflight(args,envelope=False):
    assert args.arm in ARMS and sha(args.plan)==args.expected_plan_sha256
    plan=read(args.plan)
    from backend_gate import validate_backend_binding
    validate_backend_binding(plan,allow_unbound=bool(getattr(args,'check_only',False)))
    assert plan['schema']==SCHEMA and plan['arm_order']==list(ARMS) and plan['configuration']==CONFIG
    assert plan['method_configuration']=={'method': 'InfLLM', 'runtime_arm': 'infllm_resident_qwen3', 'adapter_active': False, 'upstream_commit': '12b70798f56e56ebb23c53c7018091a3f540a028', 'KV_dtype': 'float16', 'KV_backing': 'persistent_GPU', 'attention': 'pinned_upstream_Torch_multistage_joint_softmax', 'Qwen3_qk_normalization': 'preserved_before_InfLLM_RoPE', 'prompt_truncation': False, 'same_SDPA_primitive': False}
    assert plan['backbone_dtype']=='float16' and plan['attention_backend']=='infllm_upstream_torch_joint_softmax'
    assert not plan['training_or_runtime_offload'] and not plan['automatic_retry']
    assert plan['resource_policy']==RESOURCE and plan['python']==PYTHON
    for name,digest in plan['source_sha256'].items():assert sha(local(name))==digest,name
    for key in ('fixture','activation','natural_qa_adapter'):
        binding=plan[key];assert sha(local(binding['path']))==binding['sha256'],key
    for binding in plan['qualification_bindings'].values():assert sha(local(binding['path']))==binding['sha256']
    fixture=load_fixture(plan);items=fixture['items'];groups=document_groups(fixture)
    assert len(items)==len({r['id'] for r in items})==plan['items_per_arm']==N
    assert len(groups)==plan['documents_per_arm']==D
    assert plan['expected_complete_phases_per_arm']==plan['expected_total_phases']==4*D+2*N+1==PHASES
    assert plan['expected_total_answers']==N and plan['expected_total_Writes']==D
    for row in items:
        assert not ({'answer','answers','gold','reference','references','outputs','all_classes','evidence','category','source_qa','is_abstention'}&set(row))
        assert row['dataset']==DATASET and row['max_new_tokens']==CONFIG['max_new_tokens']
        assert row['prefix_token_ids']==[151643] and row['eos_token_id']==151645
        assert row['document_token_ids'] and row['query_token_ids'] and row['bare_question_token_ids']
        assert all(type(t) is int and 0<=t<151936 for key in ('document_token_ids','query_token_ids','bare_question_token_ids') for t in row[key])
        full=[151643]+row['document_token_ids']+row['query_token_ids']
        assert token_sha(full)==row['full_prompt_with_BOS_token_sha256']
        assert len(full)+CONFIG['max_new_tokens']<=plan['full_input_coverage_maximum']
    assert max(1+len(r['document_token_ids'])+len(r['query_token_ids'])+CONFIG['max_new_tokens'] for r in items)==plan['full_input_coverage_maximum']
    if plan['full_input_coverage_maximum']>40960:
        assert plan['unscaled_full_context_stress'] and not plan['YaRN_or_RoPE_extension']
        for key in ('input_coverage','high_position_CPU_qualification'):
            if key in plan:
                binding=plan[key];assert sha(local(binding['path']))==binding['sha256']
    activation=read(local(plan['activation']['path']))
    assert activation['model_file_sha256']==plan['exact_model_file_sha256']
    assert Path(args.output).resolve()==local(plan['outputs'][args.arm])
    out=Path(args.output)
    if envelope:
        record=read(out/'execution.json')
        assert record['plan_sha256']==args.expected_plan_sha256 and record['arm']==args.arm
        assert not(out/'worker.json').exists() and not(out/'result.json').exists()
    else:assert not out.exists(),'No duplicate output/resume'
    resolve_driver(plan)
    assert not any(name in sys.modules for name in ('torch','transformers','peft')) if getattr(args,'check_only',False) else True
    return plan,activation
