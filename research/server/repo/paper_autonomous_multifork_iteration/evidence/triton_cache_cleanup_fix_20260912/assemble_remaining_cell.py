"""Seven bounded sibling packages; exact prior scientific sources plus frozen cache hook."""
import argparse,ast,copy,datetime,difflib,hashlib,json,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SNAPSHOT=HERE/'remaining_candidate_snapshot.json'
REMOTE='/srv/encbank/qcomem_align_codex_20260911/repo'
MODULE_SHA='273eb77a08c2f6b9b5dad9436e4b4bcc884514b60983a8d065e6ac5327447715'
HOOK="  from qcomem_triton_cache import install as install_triton_cache\n  record['triton_cache_cleanup_fix']=install_triton_cache(plan['task_cache_root']+'/triton')\n  save(output/'worker.json',record)\n"
PROVENANCE_HOOK="   provenance['triton_cache_cleanup_fix']=record['triton_cache_cleanup_fix']\n"
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
def config(cell):return next(x for x in read(SNAPSHOT)['ordered_candidates'] if x['cell']==cell)
def renamed(value):return value.replace('_admission_fixed','_cache_fixed') if '_admission_fixed' in value else value.replace('_attempt1','_cache_fixed_attempt1')
def job(plan):return plan['job_name'] if 'job_name' in plan else plan['scheduler']['job_name']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True,choices=[x['cell'] for x in read(SNAPSHOT)['ordered_candidates']]);cell=ap.parse_args().cell
 c=config(cell);original=c['original'];BASE=(ROOT/original['plan']['path']).parent;OLD=BASE.parent;H=ROOT/c['new_directory'];P=H/'package'
 assert len(H.relative_to(ROOT).parts)==len(OLD.relative_to(ROOT).parts),'Keep proven repository-parent depths unchanged'
 for key in ('readiness','plan','manifest','dispatcher','actual_CPU_receipt','actual_CPU_report','resource_guard'):
  s=original[key];assert sha(ROOT/s['path'])==s['sha256'],(cell,key)
 prior=read(ROOT/original['actual_CPU_receipt']['path']);assert prior['actual_exit_code']==0 and prior['actual_parent_wait']
 old=read(BASE/'plan.json');plan=copy.deepcopy(old);fix=read(HERE/'provenance.json')
 assert sha(HERE/'qcomem_triton_cache.py')==MODULE_SHA==fix['fix_module']['sha256']
 actual=read(HERE/'actual_remote_CPU_check_attempt2/stdout.json');actual_exit=read(HERE/'actual_remote_CPU_check_attempt2/execution_receipt.json')
 assert actual['status']=='PASS_actual_Triton36_CPU_install_factory_and_binary_text_put' and actual['module_sha256']==MODULE_SHA
 assert actual_exit['actual_exit_code']==0 and actual_exit['actual_parent_wait']
 for r,d in old['source_sha256'].items():assert sha(ROOT/r)==d,r
 for s in read(BASE/'upload_manifest.json')['files']:assert sha(ROOT/s['relative_path'])==s['sha256'],s['relative_path']
 old_job=job(old);new_job=old_job.replace('-admfix','-cachefix') if old_job.endswith('-admfix') else old_job.replace('-codex','-cachefix-codex')
 assert old_job!=new_job
 cache=renamed(old['task_cache_root']);assert cache!=old['task_cache_root']
 tree=ast.parse((OLD/'remote_ops.py').read_text('utf-8'))
 old_control=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONTROL' for t in n.targets))
 control=renamed(old_control);assert control!=old_control
 def namespace(t):return t.replace(rel(BASE),rel(P)).replace(old_job,new_job).replace(old['task_cache_root'],cache).replace(old_control,control)
 P.mkdir(parents=True,exist_ok=False);sources={r:d for r,d in old['source_sha256'].items() if not r.startswith(rel(BASE)+'/')};deltas=[]
 for src in sorted(BASE.glob('*.py')):
  before=src.read_text('utf-8');after=namespace(before)
  if src.name=='resource_worker.py':
   anchor='  import torch,transformers,peft,triton\n';assert after.count(anchor)==1
   after=after.replace(anchor,HOOK+anchor)
   anchor="   provenance['backend_qualification']=plan['backend_qualification']\n";assert after.count(anchor)==1
   after=after.replace(anchor,PROVENANCE_HOOK+anchor)
  write(P/src.name,after);sources[rel(P/src.name)]=sha(P/src.name)
  deltas.append({'base':bind(src),'new':bind(P/src.name),'byte_identical':src.read_bytes()==(P/src.name).read_bytes()})
  if before!=after:write(H/'source_diffs'/(src.name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(src),tofile=rel(P/src.name))))
 write(P/'qcomem_triton_cache.py',(HERE/'qcomem_triton_cache.py').read_text('utf-8'));assert sha(P/'qcomem_triton_cache.py')==MODULE_SHA
 sources[rel(P/'qcomem_triton_cache.py')]=MODULE_SHA
 plan.update(frozen_at=datetime.datetime.now().astimezone().isoformat(),source_sha256=sources,batch_output=rel(P/'run/attempt1'),task_cache_root=cache)
 if 'job_name' in old:plan['job_name']=new_job
 else:plan['scheduler']['job_name']=new_job
 plan['outputs']={a:plan['batch_output']+'/'+a for a in plan['arm_order']}
 plan['triton_cache_cleanup_fix']={'base_plan':original['plan'],'base_readiness':original['readiness'],'base_science_CPU_report':original['actual_CPU_report'],'base_actual_held_CPU_exit':original['actual_CPU_receipt'],
  'module':bind(P/'qcomem_triton_cache.py'),'fix_provenance':bind(HERE/'provenance.json'),'captured_actual_installed_source':fix['captured_actual_installed_source'],'focused_fault_report':fix['focused_report'],'actual_held_fault_exit':fix['actual_held_CPU_exit'],
  'actual_installed_runtime_CPU_report':bind(HERE/'actual_remote_CPU_check_attempt2/stdout.json'),'actual_installed_runtime_CPU_exit':bind(HERE/'actual_remote_CPU_check_attempt2/execution_receipt.json'),'assembler':bind(Path(__file__)),'immutable_candidate_snapshot':bind(SNAPSHOT),
  'change':'Same proven process-local cache manager before torch/model/KIVI imports; only exact retained temporary leaf EBUSY after atomic replace can be tolerated, after committed-byte verification and a structured stderr event.',
  'same_science_inputs_model_datatypes_scoring_timing_and_all_original_rows':True,'previous_frozen_packages_and_shared_environment_unmodified':True,'trigger_other_cell_job':25084,'this_cell_not_claimed_failed':True,
  'same_depth_sibling_preserves_original_repository_parent_indices':True,'event_log':'Each worker stderr.log; worker.json and result provenance retain installation receipt.'}
 if 'seed_annotation_erratum' in original:
  s=original['seed_annotation_erratum'];assert sha(ROOT/s['path'])==s['sha256'];plan['triton_cache_cleanup_fix']['separate_seed_annotation_erratum']=s
 save(P/'plan.json',plan)
 write(P/'batch.sbatch',namespace((BASE/'batch.sbatch').read_text('utf-8')).replace(sha(BASE/'plan.json'),sha(P/'plan.json')))
 manifest=read(BASE/'upload_manifest.json');paths={ROOT/x['relative_path'].replace(rel(BASE)+'/',rel(P)+'/') for x in manifest['files']};paths.add(P/'qcomem_triton_cache.py')
 for s in plan['triton_cache_cleanup_fix'].values():
  if isinstance(s,dict) and 'path' in s:paths.add(ROOT/s['path'])
 for s in fix.values():
  if isinstance(s,dict) and 'path' in s:paths.add(ROOT/s['path'])
 manifest.update(plan_sha256=sha(P/'plan.json'),task_cache_root=cache,files=[{'relative_path':rel(p),'local_path':str(p),'remote_path':REMOTE+'/'+rel(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths)])
 save(P/'upload_manifest.json',manifest)
 commands=read(OLD/'remote_commands.json');rp=REMOTE+'/'+rel(P)
 commands.update(remote_package=rp,plan_sha256=sha(P/'plan.json'),upload_manifest={'path':str(P/'upload_manifest.json'),'sha256':sha(P/'upload_manifest.json')},manifest_itself_remote_path=rp+'/upload_manifest.json',sbatch_argv=['sbatch','--parsable',rp+'/batch.sbatch'])
 commands['sbatch_command']=shlex.join(commands['sbatch_argv']);commands['roots_and_staged_check_command']='QCOMEM_REPO_ROOT='+REMOTE+' '+shlex.join([plan['python'],'-B',rp+'/validate_staged.py','--expected-manifest-sha256',sha(P/'upload_manifest.json')])
 save(H/'remote_commands.json',commands)
 for name in ('dispatch_once.py','remote_ops.py'):
  before=(OLD/name).read_text('utf-8');after=namespace(before);write(H/name,after)
  write(H/'source_diffs'/(name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(OLD/name),tofile=rel(H/name))))
 save(H/'source_delta.json',{'status':'ASSEMBLED_FOCUSED_CPU_CHECKS_PENDING','cell':cell,'label':c['label'],'base_plan':bind(BASE/'plan.json'),'new_plan':bind(P/'plan.json'),'new_manifest':bind(P/'upload_manifest.json'),'source_changes':deltas,
  'new_module':bind(P/'qcomem_triton_cache.py'),'old_control':old_control,'new_control':control,'old_job_name':old_job,'new_job_name':new_job,
  'allowed_changes':'Cache installation and provenance hook only in scientific worker; same-depth package paths, job/output/cache/control namespaces; separate fixed-module provenance. No data/method/datatype/scorer/natural timing change.',
  'no_SSH_GPU_input_generation_or_previous_frozen_file_changes':True})
 print(json.dumps({'status':'ASSEMBLED_CPU_ONLY_NOT_SUBMITTED','cell':cell,'plan':bind(P/'plan.json'),'manifest':bind(P/'upload_manifest.json')}))
if __name__=='__main__':main()
