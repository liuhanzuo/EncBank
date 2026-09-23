"""Freeze independent32K cells from proven16K launchers, preserving scientific code."""
from pathlib import Path
import argparse,ast,copy,datetime,difflib,hashlib,json,shlex
H=Path(__file__).resolve().parent.parent;ROOT=H.parents[2]
FIX=H.parent/'honly_babilong_qa1_0k_admission_fixed_20260912'
REMOTE='/srv/encbank/qencbank_align_codex_20260911/repo'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def bind(p):return {'path':rel(p),'sha256':sha(p)}
def write(p,text):
 assert not p.exists(),p
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.py':ast.parse(text)
 p.write_text(text,encoding='utf-8',newline='\n')
def save(p,x):write(p,json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def labels32(s):return s.replace('16384','32768').replace('16k','32k').replace('16K','32K')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cell',choices=['single','multikey','vt'],required=True);cell=ap.parse_args().cell
 C=H/(cell+'32k');INPUT=C/'inputs';OLD=H/(cell+'16k'+('' if cell=='single' else '_admission_fixed'));BASE=OLD/'package';P=C/'package'
 original_inputs=H/(cell+'16k')/'inputs';old=read(BASE/'plan.json');plan=copy.deepcopy(old);cap=60 if cell=='vt' else 48
 assert sha(FIX/'package/resource_guard.py')=='d0024c559d4bd0e6f0eb52c89229498decdc7a51cc0b4850ed233568560a2055'
 assert read(FIX/'focused_checks_attempt1/execution_receipt.json')['actual_exit_code']==0
 assert not P.exists();P.mkdir()
 if cell!='vt':
  receipt=read(INPUT/'assembly_attempt1/execution_receipt.json');assert receipt['actual_exit_code']==0 and receipt['actual_parent_wait']
  original=read(INPUT/'fixture.json');docs={d['document_id']:d for d in original['documents']};items=[];dataset=labels32(read(ROOT/old['fixture']['path'])['items'][0]['dataset'])
  for x in original['items']:
   items.append({'id':x['item_id'],'dataset':dataset,'document_id':x['document_id'],'source_id':x['item_id'],'source_ordinal':x['sample_index'],'document_token_ids':docs[x['document_id']]['document_token_ids'],'query_token_ids':x['query_token_ids'],'bare_question_token_ids':x['bare_question_token_ids'],'prefix_token_ids':[151643],'eos_token_id':151645,'max_new_tokens':48,'full_prompt_with_BOS_token_sha256':x['full_prompt_token_sha256']})
  save(INPUT/'inference_fixture.json',{'schema':'inference_only_RULER_'+cell+'_32k100_v1','source':bind(INPUT/'fixture.json'),'items':items})
  gold=read(INPUT/'scoring_only/labels.json');save(INPUT/'scoring_only/projected_labels.json',{'fixture_sha256':sha(INPUT/'inference_fixture.json'),'items':[{'id':x['item_id'],'references':x['references']} for x in gold['items']]})
 else:
  receipt=read(C/'input_generation_attempt1/execution_receipt.json');assert receipt['actual_exit_code']==0 and receipt['actual_parent_wait']
 oldprefix=rel(BASE);newprefix=rel(P);job='qencbank-ruler-'+cell+'32k100-codex';cache='/srv/encbank/qencbank_align_codex_20260911/task_cache/ruler_'+cell+'32k100_attempt1'
 def namespace(s):
  s=s.replace(oldprefix,newprefix).replace(old['task_cache_root'],cache).replace(old['job_name'],job)
  s=s.replace('ruler_'+cell+'16k100_admission_fixed_dispatch_attempt1','ruler_'+cell+'32k100_dispatch_attempt1').replace('ruler_'+cell+'16k100_dispatch_attempt1','ruler_'+cell+'32k100_dispatch_attempt1')
  return labels32(s)
 deltas=[];sources={k:v for k,v in old['source_sha256'].items() if not k.startswith(oldprefix+'/')}
 for src in BASE.glob('*.py'):
  before=src.read_text('utf-8');after=(FIX/'package/resource_guard.py').read_text('utf-8') if src.name=='resource_guard.py' else namespace(before)
  write(P/src.name,after);sources[rel(P/src.name)]=sha(P/src.name)
  deltas.append({'base':bind(src),'new':bind(P/src.name),'byte_identical':src.read_bytes()==(P/src.name).read_bytes(),'allowed_changes':'exact proven guard or32K schema/dataset/namespace labels only; method code unchanged'})
  if before!=after:write(C/'source_diffs'/(src.name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(src),tofile=str(P/src.name))))
 rebound={}
 for k in ('fixture','labels','input_generation','input_generation_receipt','input_assembly_receipt','input_manifest'):
  if k not in old:continue
  prior=ROOT/old[k]['path'];newpath=Path(str(prior).replace(str(H/(cell+'16k')),str(C)))
  plan[k]=bind(newpath);rebound[rel(prior)]=newpath
 plan.pop('admission_fix_provenance',None)
 plan.update(frozen_at=datetime.datetime.now().astimezone().isoformat(),schema=labels32(old['schema']),source_sha256=sources,batch_output=rel(P/'run/attempt1'),task_cache_root=cache,job_name=job)
 plan['outputs']={a:plan['batch_output']+'/'+a for a in plan['arm_order']}
 plan['task']=json.loads(labels32(json.dumps(old['task'])))
 for k in ('input_scope','preparation_scope'):
  if k in plan:plan[k]=labels32(plan[k])
 lengths=read(INPUT/'lengths.json');keys=['document_tokens','query_tokens','bare_question_tokens',('logical_prompt_plus_cap' if cell=='vt' else 'inference_full_prompt_plus_cap')]
 plan['actual_frozen_input_lengths']={k:{'minimum':min(x[k] for x in lengths),'maximum':max(x[k] for x in lengths)} for k in keys}
 plan['preparation_provenance']={'base16K_plan':bind(BASE/'plan.json'),'input_route_delta':bind(C/'input_source_delta.json'),'assembler':bind(__file__),'verified_fixed_guard':bind(FIX/'package/resource_guard.py'),'guard_regression_report':bind(FIX/'focused_checks_attempt1/report.json'),'guard_actual_held_exit':bind(FIX/'focused_checks_attempt1/execution_receipt.json'),'change':'New100-item32K cohort via same pinned task-specific CPU generator, cap48/48/60, exact fixed admission guard; model/method/scorer/statistics/infra unchanged.','not_a_rerun_of_completed_cell':True,'original_model_max_position_embeddings':40960,'rope_configuration_changed':False,'all_original_generated_rows_retained':True}
 save(P/'plan.json',plan);write(P/'batch.sbatch',namespace((BASE/'batch.sbatch').read_text('utf-8')).replace(sha(BASE/'plan.json'),sha(P/'plan.json')))
 m=read(BASE/'upload_manifest.json');paths=set()
 for x in m['files']:
  rp=x['relative_path']
  if rp in rebound:paths.add(rebound[rp])
  elif rp.startswith(oldprefix+'/'):paths.add(ROOT/rp.replace(oldprefix+'/',newprefix+'/',1))
  else:paths.add(ROOT/rp)
 for spec in plan['preparation_provenance'].values():
  if isinstance(spec,dict) and 'path' in spec:paths.add(ROOT/spec['path'])
 for k in ('fixture','labels','input_generation','input_generation_receipt','input_assembly_receipt','input_manifest','scorer','postprocess'):
  if k in plan:paths.add(ROOT/plan[k]['path'])
 m.update(plan_sha256=sha(P/'plan.json'),task_cache_root=cache,files=[{'relative_path':rel(p),'local_path':str(p),'remote_path':REMOTE+'/'+rel(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths)])
 save(P/'upload_manifest.json',m)
 commands=read(OLD/'remote_commands.json');rp=REMOTE+'/'+newprefix
 commands.update(remote_package=rp,plan_sha256=sha(P/'plan.json'),upload_manifest={'path':str(P/'upload_manifest.json'),'sha256':sha(P/'upload_manifest.json')},manifest_itself_remote_path=rp+'/upload_manifest.json',sbatch_argv=['sbatch','--parsable',rp+'/batch.sbatch'])
 commands['sbatch_command']=shlex.join(commands['sbatch_argv']);commands['roots_and_staged_check_command']='QENCBANK_REPO_ROOT='+REMOTE+' '+shlex.join([plan['python'],'-B',rp+'/validate_staged.py','--expected-manifest-sha256',sha(P/'upload_manifest.json')]);save(C/'remote_commands.json',commands)
 for name in ('dispatch_once.py','remote_ops.py'):write(C/name,namespace((OLD/name).read_text('utf-8')))
 save(C/'source_delta.json',{'status':'NEW32K_FOCUSED_CHECKS_PENDING','cell':cell+'32k','source_changes':deltas,'base_plan':bind(BASE/'plan.json'),'new_plan':bind(P/'plan.json'),'new_manifest':bind(P/'upload_manifest.json'),'input_rows':100,'source_cap':cap,'no_original_package_or_input_change':True,'no_GPU_SSH_model_execution':True})
 print(json.dumps({'cell':cell+'32k','status':'ASSEMBLED_CPU_ONLY','plan_sha256':sha(P/'plan.json'),'manifest_sha256':sha(P/'upload_manifest.json'),'length_ranges':plan['actual_frozen_input_lengths']}))
if __name__=='__main__':main()
