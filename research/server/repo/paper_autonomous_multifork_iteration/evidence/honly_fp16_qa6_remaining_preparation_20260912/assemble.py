"""Four new matched FP16 cohorts. Local CPU assembly only; no remote actions."""
import argparse, ast, copy, datetime, difflib, hashlib, io, json, re, shlex, tokenize
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; E=HERE.parent
OLD=E/'honly_narrativeqa_full_unscaled_admission_fixed_20260912'; BASE=OLD/'package'
BFROOT=E/'comem_honly_formal_20260911/next_longbench_qa6'
FIX=E/'triton_cache_cleanup_fix_20260912'
REMOTE='/srv/encbank/qcomem_align_codex_20260911/repo'
MODULE_SHA='273eb77a08c2f6b9b5dad9436e4b4bcc884514b60983a8d065e6ac5327447715'
CASES={'qasper':('qasper_full200_candidate','Qasper'), 'hotpotqa':('hotpot_full200_candidate','HotpotQA'), '2wikimqa':('twowiki_full200_candidate','2WikiMQA'), 'musique':('musique_full200_candidate','MuSiQue')}
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def bind(p):return {'path':rel(p),'sha256':sha(p)}
def write(p,t):
    assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix=='.py':ast.parse(t)
    p.write_text(t,encoding='utf-8',newline='\n')
def save(p,x):write(p,json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def numbers(t,mapping):
    ts=[]
    for tok in tokenize.generate_tokens(io.StringIO(t).readline):
        if tok.type==tokenize.NUMBER and tok.string in mapping:tok=tok._replace(string=str(mapping[tok.string]))
        ts.append(tok)
    return tokenize.untokenize(ts)
def all_bindings(x):
    if isinstance(x,dict):
        if isinstance(x.get('path'),str) and 'sha256' in x:yield x
        for v in x.values():yield from all_bindings(v)
    elif isinstance(x,list):
        for v in x:yield from all_bindings(v)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=list(CASES),required=True);ds=ap.parse_args().dataset
    cell,title=CASES[ds]; H=HERE/ds; P=H/'package'; assert not P.exists()
    bfpath=BFROOT/cell/'package/plan.json'; bf=read(bfpath); old=read(BASE/'plan.json'); plan=copy.deepcopy(old)
    for s in (bf['fixture'],bf['labels'],bf['scorer'],bf['official_eval'],old['natural_qa_adapter']):assert sha(ROOT/s['path'])==s['sha256']
    for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest,name
    fixture=read(ROOT/bf['fixture']['path']); items=fixture['items']; groups={}
    for i,row in enumerate(items):groups.setdefault(row['document_id'],[]).append(i)
    N=len(items);D=len(groups);cap=bf['configuration']['max_new_tokens'];phase=4*D+2*N+1
    assert N==bf['grouped_input_contract']['items']==200 and D==bf['grouped_input_contract']['documents']
    assert {x['dataset'] for x in items}=={ds} and {x['max_new_tokens'] for x in items}=={cap}
    assert bf['bootstrap']['seed']==bf['analysis']['seed']==20260911
    assert bf['bootstrap']['draws']==bf['analysis']['draws']==10000 and bf['bootstrap']['endpoint_indices']==[249,9749]
    lengths=[{'id':x['id'],'document_tokens':len(x['document_token_ids']),'query_tokens':len(x['query_token_ids']),'bare_question_tokens':len(x['bare_question_token_ids']),'full_prompt_tokens_with_BOS':1+len(x['document_token_ids'])+len(x['query_token_ids']),'prompt_plus_cap':1+len(x['document_token_ids'])+len(x['query_token_ids'])+cap} for x in items]
    maximum=max(x['prompt_plus_cap'] for x in lengths)
    assert maximum<=40960,'These completed cohorts must retain their original in-window scope'
    coverage={'status':'EXACT_ORIGINAL_BF16_COHORT_INPUTS_REUSED_WITHOUT_REENCODING','N':N,'D':D,'max_prompt_plus_cap':maximum,'model_config_max_position_embeddings':40960,'rope_theta':1000000,'rope_scaling':None,'YaRN':False,'fixture':bf['fixture'],'original_BF16_plan':bind(bfpath),'items':lengths}
    save(H/'input_coverage.json',coverage)
    job='qcomem-'+ds+'200-fp16-cachefix-codex';cache='/srv/encbank/qcomem_align_codex_20260911/task_cache/'+ds+'200_fp16_cache_fixed_attempt1'
    control='/srv/encbank/qcomem_align_codex_20260911/'+ds+'200_fp16_dispatch_cache_fixed_attempt1'
    oldcontrol='/srv/encbank/qcomem_align_codex_20260911/narrativeqa200_unscaled_dispatch_admission_fixed_attempt1'
    schema='full_'+ds+'200_FP16_six_method_remote_quality_v1'
    def namespace(t):return t.replace(rel(BASE),rel(P)).replace(old['job_name'],job).replace(old['task_cache_root'],cache).replace(oldcontrol,control)
    def scientific(t,name):
        t=namespace(t).replace(old['schema'],schema)
        t=t.replace('NarrativeQA',title).replace("'narrativeqa'",repr(ds))
        t=t.replace('NarrativeQA200',title+'200').replace('D20',f'D{D}').replace('/20 documents',f'/{D} documents').replace('/20 docs',f'/{D} docs')
        t=t.replace('20 exact document',f'{D} exact document').replace('20 reused entries',f'{D} reused entries').replace('bound65529',f'bound{maximum}')
        t=t.replace('120_Writes_2886_phases',f'{6*D}_Writes_{6*phase}_phases')
        if name in ('protocol.py','run_quality.py','resource_worker.py','analyze.py'):
            t=numbers(t,{'20':D,'481':phase,'65529':maximum})
        if name=='protocol.py':
            t=t.replace('ROOT = HERE.parents[3]','ROOT = HERE.parents[4]')
            t=t.replace("'max_new_tokens': 128",f"'max_new_tokens': {cap}").replace("row['max_new_tokens'] == 128",f"row['max_new_tokens'] == {cap}").replace('len(full)+128',f'len(full)+{cap}')
        if name=='run_quality.py':t=t.replace('+128',f'+{cap}')
        if name=='resource_worker.py':
            t=t.replace("'max_new_tokens':128",f"'max_new_tokens':{cap}")
            anchor='  import torch,transformers,peft,triton\n';assert t.count(anchor)==1
            t=t.replace(anchor,"  from qcomem_triton_cache import install as install_triton_cache\n  record['triton_cache_cleanup_fix']=install_triton_cache(plan['task_cache_root']+'/triton')\n  save(output/'worker.json',record)\n"+anchor)
            anchor="   provenance['backend_qualification']=plan['backend_qualification']\n";assert t.count(anchor)==1
            t=t.replace(anchor,"   provenance['triton_cache_cleanup_fix']=record['triton_cache_cleanup_fix']\n"+anchor)
        if name=='analyze.py':t=numbers(t,{'128':cap,'20260912':20260911})
        return t
    sources={r:d for r,d in old['source_sha256'].items() if not r.startswith(rel(BASE)+'/')};deltas=[]
    for src in sorted(BASE.glob('*.py')):
        before=src.read_text('utf-8');after=scientific(before,src.name);write(P/src.name,after);sources[rel(P/src.name)]=sha(P/src.name)
        deltas.append({'base':bind(src),'new':bind(P/src.name),'byte_identical':src.read_bytes()==(P/src.name).read_bytes()})
        if before!=after:write(H/'source_diffs'/(src.name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(src),tofile=rel(P/src.name))))
    write(P/'qcomem_triton_cache.py',(FIX/'qcomem_triton_cache.py').read_text('utf-8'));assert sha(P/'qcomem_triton_cache.py')==MODULE_SHA
    sources[rel(P/'qcomem_triton_cache.py')]=MODULE_SHA
    for key in ('fixture','labels','scorer','official_eval'):plan[key]=copy.deepcopy(bf[key])
    plan.update(schema=schema,frozen_at=datetime.datetime.now().astimezone().isoformat(),source_sha256=sources,batch_output=rel(P/'run/attempt1'),task_cache_root=cache,job_name=job,items_per_arm=N,documents_per_arm=D,expected_complete_phases_per_arm=phase,expected_total_answers=6*N,expected_total_Writes=6*D,expected_total_phases=6*phase)
    plan['configuration']['max_new_tokens']=cap;plan['outputs']={a:plan['batch_output']+'/'+a for a in plan['arm_order']}
    plan['actual_frozen_input_lengths']={k:{'minimum':min(x[k] for x in lengths),'maximum':max(x[k] for x in lengths)} for k in lengths[0] if k!='id'}
    plan['input_scope']=f'Exact completed ordinary-test BF16 {title} fixture: all {N} questions/{D} exact context groups, original tokens/order/labels/cap{cap} byte-identical. New six-method FP16 comparison, no BF16 rerun or score reuse.'
    plan['analysis'].update(metric=f'Official LongBench {title} maximum-reference QA F1 and official per-dataset scorer, untrimmed predictions',cluster_unit=f'{D} exact-context groups/{N} items; item-weighted paired context bootstrap',seed=bf['bootstrap']['seed'],draws=bf['bootstrap']['draws'],zero_based_endpoints=bf['bootstrap']['endpoint_indices'])
    plan['input_coverage']=bind(H/'input_coverage.json');plan['full_input_coverage_maximum']=maximum
    plan['task']={'benchmark':'LongBench','dataset':ds,'items':N,'documents':D,'cap':cap,'source_ordinals':[min(x['source_ordinal'] for x in items),max(x['source_ordinal'] for x in items)],'max_prompt_tokens':max(x['full_prompt_tokens_with_BOS'] for x in lengths),'max_prompt_plus_cap':maximum,'drop_long_inputs':False}
    for key in ('input_manifest','input_projection_receipt','admission_fix_provenance'):plan.pop(key,None)
    plan['preparation_scope']='New explicitly matched FP16 full cohort; root submits with fresh max-four request and duplicate checks. No SSH/GPU/smoke in preparation.'
    plan['not_claimed']=['new_quality_from_preparation','BF16_KIVI_completion','QA6_macro_until_all_six_common_FP16_datasets_close','fixed_work_infra_or_cross_device_speed_ranking','quantization_benefit_from_whole_method_Dense_KIVI_contrasts']
    plan['original_cohort']={'plan':bind(bfpath),'fixture':bf['fixture'],'labels':bf['labels'],'original_bootstrap':bf['bootstrap'],'all_original_rows_and_empty_zero_cap_results_retained':True}
    plan['method_lineage']={'Narrative_FP16_plan':bind(BASE/'plan.json'),'actual_bound_natural_reader':old['natural_qa_adapter'],'Narrative_reader_binding_correction':bind(OLD/'completion_preparation/binding_correction.json'),'all_original_tokenwise_runtime_source_hashes_unchanged':True,'MFQA_FP16_already_complete_do_not_rerun':True,'Narrative_FP16_job_25168_active_do_not_duplicate':True,'scientific_changes_vs_Narrative':'Dataset, full cohort cardinality, official dataset cap, original BF16 bootstrap seed20260911; identical six FP16 methods, full-query natural reader, model/adapter/runtime/timing/position measurement.'}
    fix=read(FIX/'provenance.json');actual=read(FIX/'actual_remote_CPU_check_attempt2/stdout.json');ex=read(FIX/'actual_remote_CPU_check_attempt2/execution_receipt.json')
    assert actual['module_sha256']==MODULE_SHA and actual['status']=='PASS_actual_Triton36_CPU_install_factory_and_binary_text_put' and ex['actual_exit_code']==0 and ex['actual_parent_wait']
    plan['triton_cache_cleanup_fix']={'module':bind(P/'qcomem_triton_cache.py'),'fix_provenance':bind(FIX/'provenance.json'),'actual_installed_runtime_CPU_report':bind(FIX/'actual_remote_CPU_check_attempt2/stdout.json'),'actual_installed_runtime_CPU_exit':bind(FIX/'actual_remote_CPU_check_attempt2/execution_receipt.json'),'change':'Exact proven process-local cache manager before model/KIVI imports. Only verified committed bytes after atomic replace permit exact temporary-leaf EBUSY cleanup tolerance, structured stderr event retained.','shared_installed_code_unmodified':True}
    save(P/'plan.json',plan)
    write(P/'batch.sbatch',namespace((BASE/'batch.sbatch').read_text('utf-8')).replace(sha(BASE/'plan.json'),sha(P/'plan.json')))
    # Retain proven backend dependency closure, replace cohort-bound fixture/labels and package.
    paths={ROOT/x['relative_path'].replace(rel(BASE)+'/',rel(P)+'/') for x in read(BASE/'upload_manifest.json')['files'] if not ('honly_narrativeqa_full_unscaled_preparation_20260912/inputs/' in x['relative_path'])}
    paths.update(P.glob('*.py'));paths.update(ROOT/r for r in plan['source_sha256'])
    paths.update(ROOT/s['path'] for s in all_bindings(plan));paths.add(Path(__file__))
    for path in paths:assert path.is_file(),path
    manifest={k:v for k,v in read(BASE/'upload_manifest.json').items() if k!='files'}
    manifest.update(plan_sha256=sha(P/'plan.json'),task_cache_root=cache,scope='Sparse source/binary/backend qualification and original cohort bindings; no model weights or generated quality results.',files=[{'relative_path':rel(x),'local_path':str(x),'remote_path':REMOTE+'/'+rel(x),'sha256':sha(x),'bytes':x.stat().st_size} for x in sorted(paths)])
    save(P/'upload_manifest.json',manifest)
    rp=REMOTE+'/'+rel(P);commands=read(OLD/'remote_commands.json')
    commands.update(remote_package=rp,plan_sha256=sha(P/'plan.json'),upload_manifest={'path':str(P/'upload_manifest.json'),'sha256':sha(P/'upload_manifest.json')},manifest_itself_remote_path=rp+'/upload_manifest.json',sbatch_argv=['sbatch','--parsable',rp+'/batch.sbatch'],scope=plan['preparation_scope'])
    commands['sbatch_command']=shlex.join(commands['sbatch_argv']);commands['roots_and_staged_check_command']='QCOMEM_REPO_ROOT='+REMOTE+' '+shlex.join([plan['python'],'-B',rp+'/validate_staged.py','--expected-manifest-sha256',sha(P/'upload_manifest.json')])
    save(H/'remote_commands.json',commands)
    for name in ('dispatch_once.py','remote_ops.py'):
        before=(OLD/name).read_text('utf-8');after=namespace(before)
        if name=='dispatch_once.py':
            after=after.replace('ROOT=HERE.parents[2]','ROOT=HERE.parents[3]').replace("'documents_per_arm':20",f"'documents_per_arm':{D}").replace("'generation_cap':128",f"'generation_cap':{cap}")
            after=after.replace('Full NarrativeQA200/20 docs six FP16 unscaled arms',f'Full {title}200/{D} docs six FP16 arms')
        write(H/name,after);write(H/'source_diffs'/(name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(OLD/name),tofile=rel(H/name))))
    save(H/'source_delta.json',{'dataset':ds,'status':'ASSEMBLED_FOCUSED_CHECKS_PENDING','original_BF16_plan':bind(bfpath),'scientific_template_plan':bind(BASE/'plan.json'),'changes':deltas,'plan':bind(P/'plan.json'),'manifest':bind(P/'upload_manifest.json'),'N':N,'D':D,'cap':cap,'expected_answers':6*N,'expected_Writes':6*D,'expected_phases':6*phase,'repository_parent_indices_adjusted_for_exact_new_package_depth':True,'only_new_namespace_written':True,'no_SSH_GPU_or_submission':True})
    print(json.dumps({'status':'ASSEMBLED_CPU_ONLY','dataset':ds,'N':N,'D':D,'cap':cap,'plan':bind(P/'plan.json'),'manifest':bind(P/'upload_manifest.json')}))
if __name__=='__main__':main()
