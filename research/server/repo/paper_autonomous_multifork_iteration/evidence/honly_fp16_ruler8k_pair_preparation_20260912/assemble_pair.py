"""CPU-only preparation of precision-matched 8K RULER; never invokes remote tools."""
import ast, copy, datetime, difflib, hashlib, json, shlex
from pathlib import Path

H=Path(__file__).resolve().parent; ROOT=H.parents[2]; E=H.parent
OLD=E/'honly_ruler_single32k_cache_fixed_20260912'; BASE=OLD/'package'
REMOTE='/srv/encbank/qencbank_align_codex_20260911/repo'
CACHE_SHA='273eb77a08c2f6b9b5dad9436e4b4bcc884514b60983a8d065e6ac5327447715'
CELLS=[('single8k','single2','niah_single_2',E/'encbank_honly_formal_20260911/remote_formal/plan.json',2,1),
       ('multikey8k','multikey1','niah_multikey_1',E/'honly_ruler_multikey_preparation_20260912/package/plan.json',1,4)]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
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
def frozen_copy(src,dst):
 assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes());assert sha(src)==sha(dst)
def specs(x):
 if isinstance(x,dict):
  if set(x)=={'path','sha256'}:yield x
  else:
   for v in x.values():yield from specs(v)
 elif isinstance(x,list):
  for v in x:yield from specs(v)

ADAPTER='''
def load_fixture(plan):
    """Read byte-identical legacy token fixture and derive non-scientific row metadata only.

    No labels, tokenizer, generator, token edits, sorting, filtering, or random draws.
    source_id denotes the unchanged original item ID; source_ordinal is its frozen row index.
    """
    path=local(plan['fixture']['path']);assert sha(path)==plan['fixture']['sha256']
    fixture=read(path)
    assert fixture['schema']=='inference_only_ruler_100_projection_v1'
    assert fixture['source']==plan['original_fixture']
    assert len(fixture['items'])==100
    for index,row in enumerate(fixture['items']):
        assert set(row)=={'id','document_id','document_token_ids','query_token_ids','bare_question_token_ids'}
        assert row['id']==plan['fixture_adapter']['item_id_prefix']+f'{index:03d}'
        assert row['document_id']==row['id']+'_document'
        row.update(dataset=plan['fixture_adapter']['dataset'],source_id=row['id'],source_ordinal=index,
                   prefix_token_ids=[151643],eos_token_id=151645,max_new_tokens=48,
                   full_prompt_with_BOS_token_sha256=token_sha([151643]+row['document_token_ids']+row['query_token_ids']))
    return fixture

'''

def main():
 old=read(BASE/'plan.json'); prior=read(OLD/'readiness.json')
 assert sha(BASE/'plan.json')==prior['plan_sha256'] and sha(BASE/'upload_manifest.json')==prior['manifest_sha256']
 assert read(OLD/'focused_checks_attempt1/execution_receipt.json')['actual_exit_code']==0
 assert sha(BASE/'qencbank_triton_cache.py')==CACHE_SHA
 for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest,name
 for cell,short,task,prior_path,gen_attempt,needles in CELLS:
  C=H/cell; P=C/'package'; I=C/'inputs'; P.mkdir(parents=True,exist_ok=False)
  legacy=read(prior_path); cohort=prior_path.parent.parent; original=ROOT/legacy['original_fixture']['path']
  fixture=ROOT/legacy['fixture']['path']; labels=ROOT/legacy['labels']['path']
  for key in ('fixture','original_fixture','labels','scorer','postprocess','activation'):
   assert sha(ROOT/legacy[key]['path'])==legacy[key]['sha256'],key
  assert legacy['activation']==old['activation'] and legacy['model_root']==old['model_root'] and legacy['adapter_root']==old['adapter_root']
  generation=cohort/'inputs/generation_plan.json';gen_receipt=cohort/f'inputs/generation_attempt{gen_attempt}/execution_receipt.json'; assembly=cohort/'inputs/assembly_attempt1/execution_receipt.json'
  assert read(generation)['random_seed']==42 and read(generation)['official_max_seq_length']==8192
  assert read(gen_receipt)['actual_exit_code']==read(assembly)['actual_exit_code']==0
  frozen_copy(fixture,I/'inference_fixture.json');frozen_copy(labels,I/'scoring_only/labels.json')
  job=f'qencbank-ruler-{short}-8k100-fp16-codex'
  cache='/srv/encbank/qencbank_align_codex_20260911/task_cache/ruler_'+short+'_8k100_fp16_attempt1'
  control='ruler_'+short+'_8k100_fp16_dispatch_attempt1'
  def adapt(text):
   return (text.replace(rel(BASE),rel(P)).replace(old['job_name'],job).replace(old['task_cache_root'],cache)
           .replace('ruler_single32k100_cache_fixed_dispatch_attempt1',control)
           .replace('ruler_niah_single_2_32k','ruler_'+task+'_8k')
           .replace('RULER_single2_32K','RULER_'+short+'_8K')
           .replace('single2_32K',''+short+'_8K').replace('single2 32K',short+' 8K'))
  sources={r:d for r,d in old['source_sha256'].items() if not r.startswith(rel(BASE)+'/')};delta=[]
  for src in BASE.glob('*.py'):
   before=src.read_text('utf-8');after=adapt(before)
   if src.name=='protocol.py':
    after=after.replace('ROOT = HERE.parents[3]','ROOT = HERE.parents[4]')
    anchor='def document_groups(fixture):';assert after.count(anchor)==1;after=after.replace(anchor,ADAPTER+anchor)
    after=after.replace("fixture = read(local(plan['fixture']['path']))",'fixture = load_fixture(plan)')
   if src.name=='resource_worker.py':
    after=after.replace('save,sha,resolve_driver','save,sha,resolve_driver,load_fixture')
    after=after.replace("fixture=read(local(plan['fixture']['path']))",'fixture=load_fixture(plan)')
   if src.name=='analyze.py':
    after=after.replace('document_groups, token_sha','document_groups, token_sha, load_fixture')
    after=after.replace("fixture=read(local(plan['fixture']['path']))",'fixture=load_fixture(plan)')
    after=after.replace("labels['fixture_sha256']==plan['fixture']['sha256']","labels['fixture_sha256']==plan['original_fixture']['sha256']")
    after=after.replace("label_map={x['id']:x for x in labels['items']}","label_map={x['item_id']:x for x in labels['items']}")
   write(P/src.name,after);sources[rel(P/src.name)]=sha(P/src.name)
   delta.append({'base':bind(src),'new':bind(P/src.name),'byte_identical':src.read_bytes()==(P/src.name).read_bytes()})
   if before!=after:write(C/'source_diffs'/(src.name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(src),tofile=rel(P/src.name))))
  assert sha(P/'qencbank_triton_cache.py')==CACHE_SHA
  plan=copy.deepcopy(old)
  plan.update(schema=f'RULER_{short}_8K100_FP16_six_method_remote_quality_v1',frozen_at=datetime.datetime.now().astimezone().isoformat(),
   fixture=bind(I/'inference_fixture.json'),labels=bind(I/'scoring_only/labels.json'),original_fixture=legacy['original_fixture'],source_sha256=sources,
   batch_output=rel(P/'run/attempt1'),task_cache_root=cache,job_name=job,
   input_generation=bind(generation),input_generation_receipt=bind(gen_receipt),input_assembly_receipt=bind(assembly),input_manifest=bind(cohort/'inputs/input_manifest.json'),
   input_scope='Original qualified BF16 '+task+' 8K 100/100 cohort, preserved byte-for-byte; new six-FP16 matched-precision evaluation with native tokenwise SDPA and unchanged scorer. No input regeneration or old score reuse.',
   preparation_scope='Prepared next-queue FP16 precision-matched 8K cell; CPU-only preparation, no remote dispatch.',
   task={'benchmark':'RULER','task':task,'length':'8k','max_seq_length':8192,'source_generation_reserve':128,'generation_cap':48,'items':100,'documents':100,'num_needle_k':needles,'num_needle_v':1,'num_needle_q':1,'input_seed':42},
   fixture_adapter={'dataset':'ruler_'+task+'_8k','item_id_prefix':'ruler_'+task+'_8k_seed42_',
     'policy':'Read original byte-identical five-field inference fixture; derive dataset/source_id(original item ID)/source_ordinal(original row index)/BOS/EOS/cap/full-token hash in memory; no label access, token edits, generator or RNG calls.'})
  plan['outputs']={arm:plan['batch_output']+'/'+arm for arm in plan['arm_order']}
  lengths=read(cohort/'inputs/lengths.json')
  plan['actual_frozen_input_lengths']={k:{'minimum':min(x[k] for x in lengths),'maximum':max(x[k] for x in lengths)} for k in ('document_tokens','query_tokens','bare_question_tokens','inference_full_prompt_plus_cap')}
  plan['analysis']['metric']='Pinned NVIDIA RULER string_match_all and unchanged official postprocess_pred; same original 8K cohort and scorer, new six-FP16 results only.'
  plan['preparation_provenance']={'runtime_base_plan':bind(BASE/'plan.json'),'runtime_base_readiness':bind(OLD/'readiness.json'),
   'prior_BF16_plan':bind(prior_path),'prior_BF16_completion':bind(cohort/'completion_registration.json'),
   'original_inference_fixture':legacy['fixture'],'original_labels':legacy['labels'],'original_full_fixture':legacy['original_fixture'],
   'assembler':bind(Path(__file__)),'prior_BF16_precision_is_not_FP16':True,'all_original_100_rows_retained':True,'inputs_and_labels_copied_byte_for_byte':True,
   'input_generation_seed_unchanged':42,'prior_BF16_bootstrap_seed':legacy['analysis']['seed'],'new_FP16_bootstrap_seed':20260912,
   'bootstrap_choice':'Freeze seed20260912 from the existing six-FP16 runtime before new results; MT19937,10000 draws,100 exact paired document clusters,endpoints249/9749 unchanged. Input seed42 is separate and unchanged.',
   'no_completed_FP16_cell_rerun':True,'original_model_max_position_embeddings':40960,'rope_configuration_changed':False,
   'allowed_delta':'Task/schema/namespace strings, repository parent depth, stdlib in-memory legacy fixture metadata adapter, original label item_id and full-fixture hash binding. Sixarm method/model/attention/cache/timing/guard/scorer APIs unchanged.'}
  plan['triton_cache_cleanup_fix']['module']=bind(P/'qencbank_triton_cache.py')
  save(P/'plan.json',plan)
  write(P/'batch.sbatch',adapt((BASE/'batch.sbatch').read_text('utf-8')).replace(sha(BASE/'plan.json'),sha(P/'plan.json')))
  # Preserve the known working sparse route. The original manifest also supplies backend proof closure.
  paths={ROOT/x['relative_path'] for x in read(BASE/'upload_manifest.json')['files']}
  paths.update(ROOT/s['path'] for s in specs(plan));paths.update(ROOT/r for r in sources)
  paths.update([P/'plan.json',P/'batch.sbatch',BASE/'upload_manifest.json'])
  im=read(cohort/'inputs/input_manifest.json')
  paths.update(ROOT/s['path'] for s in im.get('sources',im.get('bindings',[])))
  manifest={'plan_sha256':sha(P/'plan.json'),'launch_permitted':True,'remote_root':REMOTE,'model_root':plan['model_root'],'adapter_root':plan['adapter_root'],'task_cache_root':cache,
   'files':[{'relative_path':rel(p),'local_path':str(p),'remote_path':REMOTE+'/'+rel(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths)]}
  save(P/'upload_manifest.json',manifest)
  commands=read(OLD/'remote_commands.json');rp=REMOTE+'/'+rel(P)
  commands.update(remote_package=rp,plan_sha256=sha(P/'plan.json'),upload_manifest={'path':str(P/'upload_manifest.json'),'sha256':sha(P/'upload_manifest.json')},manifest_itself_remote_path=rp+'/upload_manifest.json',sbatch_argv=['sbatch','--parsable',rp+'/batch.sbatch'])
  commands['sbatch_command']=shlex.join(commands['sbatch_argv'])
  commands['roots_and_staged_check_command']='QENCBANK_REPO_ROOT='+REMOTE+' '+shlex.join([plan['python'],'-B',rp+'/validate_staged.py','--expected-manifest-sha256',sha(P/'upload_manifest.json')])
  save(C/'remote_commands.json',commands)
  for name in ('dispatch_once.py','remote_ops.py'):
   before=(OLD/name).read_text('utf-8');after=adapt(before)
   if name=='dispatch_once.py':after=after.replace('ROOT=HERE.parents[2]','ROOT=HERE.parents[3]')
   write(C/name,after)
   write(C/'source_diffs'/(name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(OLD/name),tofile=rel(C/name))))
  save(C/'source_delta.json',{'status':'ASSEMBLED_CPU_ONLY_CHECKS_PENDING','base_plan':bind(BASE/'plan.json'),'new_plan':bind(P/'plan.json'),'new_manifest':bind(P/'upload_manifest.json'),'sources':delta,
   'allowed_delta':plan['preparation_provenance']['allowed_delta'],'old_inference_fixture':legacy['fixture'],'new_identical_fixture':plan['fixture'],'old_labels':legacy['labels'],'new_identical_labels':plan['labels'],
   'no_SSH_GPU_submission_global_state_queue_plan_or_existing_frozen_files_modified':True})
  print(json.dumps({'status':'ASSEMBLED_CPU_ONLY_NOT_SUBMITTED','cell':cell,'plan':bind(P/'plan.json'),'manifest':bind(P/'upload_manifest.json')}))
if __name__=='__main__':main()
