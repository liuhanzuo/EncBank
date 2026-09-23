"""Freeze new long RULER inputs into the working six-arm cache-fixed CLI."""
import argparse,ast,copy,datetime,difflib,hashlib,json,re,shlex
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2];E=H.parent
R=E/'honly_ruler_parallel_preparation_20260912';LE=E/'honly_longeval128k_cache_fixed_20260912/package'
REMOTE='/srv/encbank/qcomem_align_codex_20260911/repo'
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
def specs(x):
 if isinstance(x,dict):
  if set(x)=={'path','sha256'}:yield x
  else:
   for v in x.values():yield from specs(v)
 elif isinstance(x,list):
  for v in x:yield from specs(v)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True);cell=ap.parse_args().cell
 m=re.fullmatch('(single|multikey|vt)(64|128)k',cell);assert m
 kind=m[1];length=int(m[2]);cap=60 if kind=='vt' else 48;C=H/cell;I=C/'inputs';P=C/'package'
 OLD=E/'honly_ruler_single32k_cache_fixed_20260912' if kind=='single' else R/(kind+'32k_cache_fixed');BASE=OLD/'package';old=read(BASE/'plan.json');le=read(LE/'plan.json')
 assert sha(BASE/'plan.json')==read(OLD/'readiness.json')['plan_sha256']
 assert read(OLD/'completion_registration.json')['status'].startswith('complete') or read(OLD/'completion_registration.json')['status'].startswith('COMPLETED')
 assert read(I/'assembly_attempt1/execution_receipt.json' if kind!='vt' else C/'input_generation_attempt1/execution_receipt.json')['actual_exit_code']==0
 P.mkdir(exist_ok=False)
 if kind!='vt':
  full=read(I/'fixture.json');docs={d['document_id']:d for d in full['documents']};items=[]
  dataset='ruler_'+old['task']['task']+'_'+str(length)+'k'
  for x in full['items']:
   items.append({'id':x['item_id'],'dataset':dataset,'document_id':x['document_id'],'source_id':x['item_id'],'source_ordinal':x['sample_index'],'document_token_ids':docs[x['document_id']]['document_token_ids'],'query_token_ids':x['query_token_ids'],'bare_question_token_ids':x['bare_question_token_ids'],'prefix_token_ids':[151643],'eos_token_id':151645,'max_new_tokens':48,'full_prompt_with_BOS_token_sha256':x['full_prompt_token_sha256']})
  save(I/'inference_fixture.json',{'schema':f'inference_only_RULER_{kind}_{length}k100_v1','source':bind(I/'fixture.json'),'items':items})
  labels=read(I/'scoring_only/labels.json');save(I/'scoring_only/projected_labels.json',{'fixture_sha256':sha(I/'inference_fixture.json'),'items':[{'id':x['item_id'],'references':x['references']} for x in labels['items']]})
 fixture=read(I/'inference_fixture.json');maximum=max(1+len(x['document_token_ids'])+len(x['query_token_ids'])+cap for x in fixture['items'])
 assert 40960<maximum<=length*1024+(61 if kind=='vt' else 0)
 job=f'qcomem-ruler-{kind}{length}k100-fp16-codex';cache=f'/srv/encbank/qcomem_align_codex_20260911/task_cache/ruler_{kind}{length}k100_fp16_attempt1';control=f'ruler_{kind}{length}k100_fp16_dispatch_attempt1'
 oldcontrol=re.search("CONTROL='([^']+)'",(OLD/'dispatch_once.py').read_text())[1]
 def adapt(t):
  t=t.replace(rel(BASE),rel(P)).replace(old['job_name'],job).replace(old['task_cache_root'],cache).replace(oldcontrol,'/srv/encbank/qcomem_align_codex_20260911/'+control)
  return t.replace('32K',str(length)+'K').replace('32k',str(length)+'k').replace('32768',str(length*1024))
 core={k:v for k,v in old['source_sha256'].items() if not k.startswith(rel(BASE)+'/') and k!=old['natural_qa_adapter']['path']}
 core[le['natural_qa_adapter']['path']]=le['natural_qa_adapter']['sha256'];sources=dict(core);deltas=[]
 qblock=(LE/'protocol.py').read_text('utf-8');qblock=qblock[qblock.index("    assert plan['full_input_original_window']"):qblock.index('    assert len(items)')]
 qblock=qblock.replace("plan['full_input_coverage_maximum']==131749",f"plan['full_input_coverage_maximum']=={maximum}")
 qblock=qblock.replace("max(17+len",f"max({cap+1}+len").replace('==131749',f'=={maximum}')
 for src in BASE.glob('*.py'):
  before=src.read_text('utf-8');after=adapt(before)
  if src.name=='protocol.py':
   after=re.sub(r'ROOT = HERE.parents\[\d\]', 'ROOT = HERE.parents[4]',after)
   anchor='    assert len(items)';assert after.count(anchor)==1;after=after.replace(anchor,qblock+anchor)
   after=after.replace(f'len(full)+{cap} <= 40960',f'len(full)+{cap} <= {maximum}')
  if src.name=='resource_worker.py':
   anchor="   model,tokenizer=load_backbone(";assert after.count(anchor)==1
   block="   original_config=read(model_path(plan,'model')/'config.json')\n   assert original_config['max_position_embeddings']==40960 and original_config['rope_theta']==1000000 and original_config.get('rope_scaling') is None and not original_config.get('use_sliding_window',False)\n"
   after=after.replace(anchor,block+anchor)
   anchor="   provenance={'plan_sha256':";assert after.count(anchor)==1
   block=f"   assert model.config.max_position_embeddings==40960\n   record['effective_model_config']=model.config.to_dict()\n   record['unscaled_input_coverage']={{'maximum_full_prompt_plus_cap':{maximum},'fixture':plan['fixture']}}\n   save(output/'worker.json',record)\n"
   after=after.replace(anchor,block+anchor)
   anchor="   provenance['node_identity']=record['node_identity']";assert after.count(anchor)==1
   block=f"   provenance['unscaled_coverage']={{'maximum_full_prompt_plus_cap':{maximum},'CPU_qualification':plan['high_position_CPU_qualification'],'effective_model_config':record['effective_model_config'],'original_max_position_embeddings':40960,'original_rope_theta':1000000,'YaRN_or_model_config_override_applied':False}}\n"
   after=after.replace(anchor,block+anchor)
  write(P/src.name,after);sources[rel(P/src.name)]=sha(P/src.name)
  deltas.append({'base':bind(src),'new':bind(P/src.name),'byte_identical':src.read_bytes()==(P/src.name).read_bytes()})
  if before!=after:write(C/'source_diffs'/(src.name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(src),tofile=rel(P/src.name))))
 plan=copy.deepcopy(old);plan.update(schema=adapt(old['schema']),frozen_at=datetime.datetime.now().astimezone().isoformat(),source_sha256=sources,fixture=bind(I/'inference_fixture.json'),labels=bind(I/'scoring_only'/('labels.json' if kind=='vt' else 'projected_labels.json')),batch_output=rel(P/'run/attempt1'),task_cache_root=cache,job_name=job,natural_qa_adapter=le['natural_qa_adapter'],full_input_coverage_maximum=maximum,high_position_CPU_qualification=le['high_position_CPU_qualification'])
 plan['task']=json.loads(adapt(json.dumps(old['task'])));plan['outputs']={a:plan['batch_output']+'/'+a for a in plan['arm_order']}
 plan['input_generation']=bind(I/'generation_plan.json');plan['input_generation_receipt']=bind(C/'input_generation_attempt1/execution_receipt.json' if kind=='vt' else I/'generation_attempt1/execution_receipt.json')
 if kind!='vt':plan['input_assembly_receipt']=bind(I/'assembly_attempt1/execution_receipt.json')
 plan['input_manifest']=bind(I/('manifest.json' if kind=='vt' else 'input_manifest.json'))
 lengths=read(I/'lengths.json');keys=['document_tokens','query_tokens','bare_question_tokens','logical_prompt_plus_cap' if kind=='vt' else 'inference_full_prompt_plus_cap'];plan['actual_frozen_input_lengths']={k:{'minimum':min(x[k] for x in lengths),'maximum':max(x[k] for x in lengths)} for k in keys}
 plan['input_scope']=f'New complete100-item pinned official {kind}{length}K cohort, seed42. No crop, no scaling/config change; full native unscaled positions beyond40960. Different examples from published paper; all generated rows retained.'
 plan['preparation_scope']='CPU-only remaining-length package; no GPU result, feasibility or quality inference from input arithmetic.'
 plan['preparation_provenance']={'base32K_plan':bind(BASE/'plan.json'),'base32K_completion':bind(OLD/'completion_registration.json'),'input_route_delta':bind(C/'input_source_delta.json'),'assembler':bind(Path(__file__)),'high_position_runtime_reference':bind(LE/'plan.json'),'passive_position_adapter':le['natural_qa_adapter'],'original_model_max_position_embeddings':40960,'rope_configuration_changed':False,'all_original_generated_rows_retained':True,'allowed_delta':'New task-length inputs and namespace; admission maximum bound to full prompt plus cap; already qualified high-position guard, original/effective config observations and passive natural-read position diagnostics. Method arithmetic/quantizers/native attention/timing/scoring/bootstrap unchanged.'}
 plan['triton_cache_cleanup_fix']['module']=bind(P/'qcomem_triton_cache.py')
 plan['scheduler_allowance']={'walltime':'24:00:00','scope':'Conservative maximum Slurm allowance for six long-input arms; not an ETA or evidence of runtime. Fresh scheduler admission required.'}
 plan['not_claimed']=list(dict.fromkeys(plan['not_claimed']+['native_pretrained128K_context','whole_model_high_position_correctness','GPU_capacity_or_feasibility_from_CPU_arithmetic','quality_from_CPU_arithmetic']))
 save(P/'plan.json',plan)
 write(P/'batch.sbatch',adapt((BASE/'batch.sbatch').read_text('utf-8')).replace(sha(BASE/'plan.json'),sha(P/'plan.json')).replace('#SBATCH --time=08:00:00','#SBATCH --time=24:00:00'))
 paths={ROOT/x['relative_path'] for x in read(BASE/'upload_manifest.json')['files'] if not x['relative_path'].startswith(rel(BASE)+'/')}
 paths.update(P.glob('*.py'));paths.update([P/'plan.json',P/'batch.sbatch']);paths.update(ROOT/r for r in sources)
 paths.update(ROOT/s['path'] for s in specs(plan));paths.update(ROOT/s['path'] for s in specs(read(I/('manifest.json' if kind=='vt' else 'input_manifest.json'))))
 # High-position registration recursively binds its exact CPU proof, never rerun.
 paths.update(ROOT/s['path'] for s in specs(read(ROOT/plan['high_position_CPU_qualification']['path'])))
 manifest=read(BASE/'upload_manifest.json');manifest.update(plan_sha256=sha(P/'plan.json'),task_cache_root=cache,files=[{'relative_path':rel(p),'local_path':str(p),'remote_path':REMOTE+'/'+rel(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths)])
 save(P/'upload_manifest.json',manifest)
 commands=read(OLD/'remote_commands.json');rp=REMOTE+'/'+rel(P)
 commands.update(remote_package=rp,plan_sha256=sha(P/'plan.json'),upload_manifest={'path':str(P/'upload_manifest.json'),'sha256':sha(P/'upload_manifest.json')},manifest_itself_remote_path=rp+'/upload_manifest.json',sbatch_argv=['sbatch','--parsable',rp+'/batch.sbatch'])
 commands['sbatch_command']=shlex.join(commands['sbatch_argv']);commands['roots_and_staged_check_command']='QCOMEM_REPO_ROOT='+REMOTE+' '+shlex.join([plan['python'],'-B',rp+'/validate_staged.py','--expected-manifest-sha256',sha(P/'upload_manifest.json')]);save(C/'remote_commands.json',commands)
 for name in ('dispatch_once.py','remote_ops.py'):
  before=(OLD/name).read_text('utf-8');after=adapt(before)
  if name=='dispatch_once.py':after=re.sub(r'ROOT=HERE.parents\[\d\]', 'ROOT=HERE.parents[3]',after)
  write(C/name,after);write(C/'source_diffs'/(name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(OLD/name),tofile=rel(C/name))))
 save(C/'source_delta.json',{'status':'ASSEMBLED_CPU_ONLY_CHECKS_PENDING','cell':cell,'base_plan':bind(BASE/'plan.json'),'new_plan':bind(P/'plan.json'),'new_manifest':bind(P/'upload_manifest.json'),'sources':deltas,'allowed_delta':plan['preparation_provenance']['allowed_delta'],'maximum_full_prompt_plus_cap':maximum,'no_GPU_SSH_submission_or_global_edits':True})
 print(json.dumps({'cell':cell,'status':'ASSEMBLED_CPU_ONLY_CHECKS_PENDING','maximum':maximum,'plan':bind(P/'plan.json')}))
if __name__=='__main__':main()
