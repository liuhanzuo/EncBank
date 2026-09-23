"""New same-input RULER single32K namespace with a process-local Triton cleanup fix."""
import ast, copy, datetime, difflib, hashlib, json, shlex
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
OLD=H.parent/'honly_ruler_parallel_preparation_20260912/single32k';BASE=OLD/'package';P=H/'package'
FIX=H.parent/'triton_cache_cleanup_fix_20260912'
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
def main():
 old=read(BASE/'plan.json');plan=copy.deepcopy(old);prior=read(OLD/'readiness.json')
 assert sha(BASE/'plan.json')==prior['plan_sha256'] and sha(BASE/'upload_manifest.json')==prior['manifest_sha256']
 assert read(OLD/'focused_checks_attempt1/execution_receipt.json')['actual_exit_code']==0
 fix=read(FIX/'provenance.json');assert fix['status']=='CPU_VERIFIED_PROCESS_LOCAL_CACHE_CLEANUP_FIX'
 assert read(FIX/'focused_checks_attempt1/execution_receipt.json')['actual_exit_code']==0
 for r,d in old['source_sha256'].items():assert sha(ROOT/r)==d,r
 for spec in read(BASE/'upload_manifest.json')['files']:assert sha(ROOT/spec['relative_path'])==spec['sha256']
 P.mkdir(exist_ok=False)
 job='qcomem-ruler-single32k100-cachefix-codex'
 cache='/srv/encbank/qcomem_align_codex_20260911/task_cache/ruler_single32k100_cache_fixed_attempt1'
 def namespace(t):return t.replace(rel(BASE),rel(P)).replace(old['job_name'],job).replace(old['task_cache_root'],cache).replace('ruler_single32k100_dispatch_attempt1','ruler_single32k100_cache_fixed_dispatch_attempt1')
 sources={r:d for r,d in old['source_sha256'].items() if not r.startswith(rel(BASE)+'/')};deltas=[]
 for src in BASE.glob('*.py'):
  before=src.read_text('utf-8');after=namespace(before)
  if src.name=='protocol.py':
   assert 'ROOT = HERE.parents[4]' in after;after=after.replace('ROOT = HERE.parents[4]','ROOT = HERE.parents[3]')
  if src.name=='resource_worker.py':
   anchor='  import torch,transformers,peft,triton\n';assert after.count(anchor)==1
   after=after.replace(anchor,"  from qcomem_triton_cache import install as install_triton_cache\n  record['triton_cache_cleanup_fix']=install_triton_cache(plan['task_cache_root']+'/triton')\n  save(output/'worker.json',record)\n"+anchor)
   anchor="   provenance.update(fixture=plan['fixture'],physical_gpu_identity=identity,resource_policy=RESOURCE,scope=";assert after.count(anchor)==1
   after=after.replace(anchor,"   provenance['triton_cache_cleanup_fix']=record['triton_cache_cleanup_fix']\n"+anchor)
  write(P/src.name,after);sources[rel(P/src.name)]=sha(P/src.name)
  deltas.append({'base':bind(src),'new':bind(P/src.name),'byte_identical':src.read_bytes()==(P/src.name).read_bytes()})
  if before!=after:write(H/'source_diffs'/(src.name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(src),tofile=rel(P/src.name))))
 write(P/'qcomem_triton_cache.py',(FIX/'qcomem_triton_cache.py').read_text('utf-8'))
 assert sha(P/'qcomem_triton_cache.py')==fix['fix_module']['sha256'];sources[rel(P/'qcomem_triton_cache.py')]=sha(P/'qcomem_triton_cache.py')
 plan.update(frozen_at=datetime.datetime.now().astimezone().isoformat(),source_sha256=sources,batch_output=rel(P/'run/attempt1'),task_cache_root=cache,job_name=job)
 plan['outputs']={a:plan['batch_output']+'/'+a for a in plan['arm_order']}
 plan['triton_cache_cleanup_fix']={'base_plan':bind(BASE/'plan.json'),'base_science_checks':bind(OLD/'focused_checks_attempt1/report.json'),'base_actual_held_CPU_exit':bind(OLD/'focused_checks_attempt1/execution_receipt.json'),
  'module':bind(P/'qcomem_triton_cache.py'),'fix_provenance':bind(FIX/'provenance.json'),'captured_actual_installed_source':fix['captured_actual_installed_source'],'focused_report':fix['focused_report'],'actual_held_CPU_exit':fix['actual_held_CPU_exit'],'assembler':bind(Path(__file__)),
  'change':'Process-local cache-manager selection before torch/model/KIVI import; only exact retained leaf EBUSY after successful atomic replace may return after committed-byte verification and structured stderr log.',
  'science_inputs_model_precision_scoring_timing_unchanged':True,'shared_environment_or_original_frozen_outputs_modified':False,
  'trigger_other_benchmark_job':25084,'this_RULER_cell_not_claimed_failed':True,'same_existing_CPU_ready_input_cohort_not_regenerated':True,
  'expected_original_Triton_version':'3.6.0','expected_original_cache_sha256':fix['captured_actual_installed_source']['sha256'],
  'event_log':'Each arm stderr.log; worker.json/result provenance include manager installation receipt.'}
 save(P/'plan.json',plan)
 write(P/'batch.sbatch',namespace((BASE/'batch.sbatch').read_text('utf-8')).replace(sha(BASE/'plan.json'),sha(P/'plan.json')))
 manifest=read(BASE/'upload_manifest.json')
 paths={ROOT/x['relative_path'].replace(rel(BASE)+'/',rel(P)+'/') for x in manifest['files']}
 paths.add(P/'qcomem_triton_cache.py')
 for spec in plan['triton_cache_cleanup_fix'].values():
  if isinstance(spec,dict) and 'path' in spec:paths.add(ROOT/spec['path'])
 for spec in fix.values():
  if isinstance(spec,dict) and 'path' in spec:paths.add(ROOT/spec['path'])
 manifest.update(plan_sha256=sha(P/'plan.json'),task_cache_root=cache,files=[{'relative_path':rel(p),'local_path':str(p),'remote_path':REMOTE+'/'+rel(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths)])
 save(P/'upload_manifest.json',manifest)
 commands=read(OLD/'remote_commands.json');rp=REMOTE+'/'+rel(P)
 commands.update(remote_package=rp,plan_sha256=sha(P/'plan.json'),upload_manifest={'path':str(P/'upload_manifest.json'),'sha256':sha(P/'upload_manifest.json')},manifest_itself_remote_path=rp+'/upload_manifest.json',sbatch_argv=['sbatch','--parsable',rp+'/batch.sbatch'])
 commands['sbatch_command']=shlex.join(commands['sbatch_argv'])
 commands['roots_and_staged_check_command']='QCOMEM_REPO_ROOT='+REMOTE+' '+shlex.join([plan['python'],'-B',rp+'/validate_staged.py','--expected-manifest-sha256',sha(P/'upload_manifest.json')])
 save(H/'remote_commands.json',commands)
 for name in ('dispatch_once.py','remote_ops.py'):
  before=(OLD/name).read_text('utf-8');after=namespace(before)
  if name=='dispatch_once.py':
   assert 'ROOT=HERE.parents[3]' in after;after=after.replace('ROOT=HERE.parents[3]','ROOT=HERE.parents[2]')
  write(H/name,after)
  write(H/'source_diffs'/(name+'.diff'),''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=rel(OLD/name),tofile=rel(H/name))))
 save(H/'source_delta.json',{'status':'NEW_CACHE_FIXED_PACKAGE_FOCUSED_CHECKS_PENDING','base_plan':bind(BASE/'plan.json'),'new_plan':bind(P/'plan.json'),'new_manifest':bind(P/'upload_manifest.json'),'sources':deltas,'new_module':bind(P/'qcomem_triton_cache.py'),
  'allowed_code_delta':'protocol repository-parent depth adjusted for new namespace; worker cache install and provenance hook; new cache module; dispatch namespace/depth only. All science, natural timing, model and input sources preserved.',
  'no_GPU_SSH_submission_or_previous_frozen_files_changed':True})
 print(json.dumps({'status':'ASSEMBLED_CPU_ONLY_NOT_SUBMITTED','plan':bind(P/'plan.json'),'manifest':bind(P/'upload_manifest.json'),'job_name':job}))
if __name__=='__main__':main()
