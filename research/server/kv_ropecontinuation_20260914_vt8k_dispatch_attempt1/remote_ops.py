"""Adapted working sparse promotion/Slurm route; sent over SSH stdin only by root."""
import argparse,ast,csv,datetime,hashlib,io,json,os,re,shlex,subprocess,tarfile
from pathlib import Path
RELATIVE='paper_autonomous_multifork_iteration/evidence/kv_ropecompat_continuation_20260914/cells/vt8k/package'
REMOTE='/srv/encbank/qcomem_align_codex_20260911/repo'
PYTHON='/srv/encbank/qcomem_runtime_20260911/python312/bin/python'
CONTROL='/srv/encbank/qcomem_align_codex_20260911/kv_ropecontinuation_20260914_vt8k_dispatch_attempt1'
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,value):
 p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def expected_files(manifest,manifest_sha):
 expected={x['relative_path']:x for x in manifest['files']}
 assert len(expected)==len(manifest['files'])
 manifest_relative=RELATIVE+'/upload_manifest.json'
 assert manifest_relative not in expected
 expected[manifest_relative]={'relative_path':manifest_relative,'remote_path':REMOTE+'/'+manifest_relative,'sha256':manifest_sha}
 return expected
def check_files(expected,root):
 for name,spec in expected.items():
  path=(root/name).resolve();path.relative_to(root)
  assert str(path)==spec['remote_path'] and path.is_file() and sha(path)==spec['sha256'],name
  if 'bytes' in spec:assert path.stat().st_size==spec['bytes']
 return len(expected)
def held(control,name,argv,env=None,cwd=None):
 with (control/(name+'.stdout')).open('xb') as out,(control/(name+'.stderr')).open('xb') as err:
  child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,cwd=cwd);code=child.wait()
 return {'argv':argv,'pid':child.pid,'actual_exit_code':code,'actual_parent_wait':True,'process_exit_observed':True,'stdout':(control/(name+'.stdout')).read_text(encoding='utf-8'),'stderr':(control/(name+'.stderr')).read_text(encoding='utf-8')}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['validate','submit'],required=True)
 for key in ('control','manifest-sha256','archive-sha256','expected-plan-sha256'):parser.add_argument('--'+key,required=True)
 a=parser.parse_args();home=Path('/srv/encbank').resolve();root=Path(REMOTE).resolve();root.relative_to(home)
 control=Path(a.control).resolve();control.relative_to(home);assert str(control)==CONTROL and control.is_dir()
 prep=read(control/'preparation_receipt.json')
 assert sha(control/'remote_ops.py')==prep['remote_ops_sha256'] and sha(control/'remote_commands.json')==prep['remote_commands_sha256']
 assert sha(control/'upload_manifest.json')==a.manifest_sha256==prep['upload_manifest_sha256']
 manifest=read(control/'upload_manifest.json');commands=read(control/'remote_commands.json');expected=expected_files(manifest,a.manifest_sha256)
 assert manifest['remote_root']==REMOTE and manifest['plan_sha256']==commands['plan_sha256']==a.expected_plan_sha256
 assert manifest['launch_permitted'] is True and commands['launch_permitted'] is True
 assert commands['remote_package']==REMOTE+'/'+RELATIVE
 here=root/RELATIVE;here.resolve().relative_to(home)
 assert commands['sbatch_argv']==['sbatch','--parsable',str(here/'batch.sbatch')]
 assert shlex.split(commands['sbatch_command'])==commands['sbatch_argv']
 if a.mode=='validate':
  report={'status':'promoting_and_validating','started_at':now(),'children':[],'GPU_or_Slurm_actions':0}
  assert not (control/'package_validation.json').exists(),'Never repeat promotion/check stage'
  save(control/'package_validation.json',report)
  try:
   assert sha(control/'sparse_package.tar')==a.archive_sha256==prep['archive_sha256']
   root.mkdir(parents=True,exist_ok=True)
   with tarfile.open(control/'sparse_package.tar','r:') as tar:
    members=tar.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
    for member in members:
     assert member.isfile() and not member.issym() and not member.islnk()
     dest=(root/member.name).resolve();dest.relative_to(root);dest.relative_to(home)
     data=tar.extractfile(member).read();spec=expected[member.name]
     assert hashlib.sha256(data).hexdigest()==spec['sha256'] and ('bytes' not in spec or len(data)==spec['bytes'])
     if dest.exists():assert sha(dest)==spec['sha256'],'Existing unequal file; no overwrite'
     else:
      dest.parent.mkdir(parents=True,exist_ok=True)
      with dest.open('xb') as stream:stream.write(data)
   report['files_verified']=check_files(expected,root)
   assert sha(here/'plan.json')==a.expected_plan_sha256;plan=read(here/'plan.json')
   assert plan['launch_permitted'] is True and plan['arm_order']==['public_kvdirect_j0']
   assert not (root/plan['batch_output']).exists()
   for name in [plan['batch_output'],*plan['outputs'].values(),*plan['source_sha256']]:
    (root/name).resolve().relative_to(home)
   for key in ('model_root','adapter_root','task_cache_root'):Path(plan[key]).resolve().relative_to(home)
   for name in expected:
    if name.endswith('.py'):ast.parse((root/name).read_text(encoding='utf-8-sig'),filename=str(root/name))
   cache=Path(plan['task_cache_root']).resolve();cache.relative_to(home)
   for leaf in ('tmp','xdg','hf','torch','triton'):(cache/leaf).mkdir(parents=True,exist_ok=True)
   env=dict(os.environ);env.update(QCOMEM_REPO_ROOT=REMOTE,TMPDIR=str(cache/'tmp'),XDG_CACHE_HOME=str(cache/'xdg'),HF_HOME=str(cache/'hf'),TORCH_HOME=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',USE_TORCH='0',USE_TF='0',USE_FLAX='0',PYTHONUTF8='1',PYTHONHASHSEED='42',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
   original_check=[PYTHON,'-B',str(here/'validate_staged.py'),'--expected-manifest-sha256',a.manifest_sha256]
   assert shlex.split(commands['roots_and_staged_check_command'])==['QCOMEM_REPO_ROOT='+REMOTE,*original_check]
   check=held(control,'original_validate_staged',original_check,env=env);report['children'].append(check);save(control/'package_validation.json',report);assert check['actual_exit_code']==0
   for arm in plan['arm_order']:
    argv=[PYTHON,'-B',str(here/'resource_worker.py'),'--plan',str(here/'plan.json'),'--expected-plan-sha256',a.expected_plan_sha256,'--arm',arm,'--output',str(root/plan['outputs'][arm]),'--check-only']
    check=held(control,'check_'+arm,argv,env=env);report['children'].append(check);save(control/'package_validation.json',report);assert check['actual_exit_code']==0
   check=held(control,'shell_syntax',['bash','-n',str(here/'batch.sbatch')],env=env);report['children'].append(check);assert check['actual_exit_code']==0
   report.update(status='PASS_sparse_files_original_staging_and_one_public_kvdirect_j0_CPU_check',finished_at=now(),resolved_storage_root=str(home),no_model_framework_or_GPU_execution=True)
  except BaseException as error:
   report.update(status='failed_no_retry',error=repr(error),finished_at=now());raise
  finally:save(control/'package_validation.json',report)
  print(json.dumps({'status':report['status'],'report_sha256':sha(control/'package_validation.json')}));return
 # A retained atomic lock disallows any retry after SSH loss or uncertain sbatch.
 (control/'submission.lock').mkdir(exist_ok=False)
 receipt=control/'submit_receipt.json';rec={'status':'fresh_validation','started_at':now(),'submission_invoked':False,'automatic_retry':False,'plan_sha256':a.expected_plan_sha256};save(receipt,rec)
 try:
  assert read(control/'package_validation.json')['status']=='PASS_sparse_files_original_staging_and_one_public_kvdirect_j0_CPU_check'
  rec['files_verified']=check_files(expected,root)
  assert sha(here/'plan.json')==a.expected_plan_sha256;plan=read(here/'plan.json')
  import sys
  sys.path.insert(0,str(here));from backend_gate import validate_backend_binding
  validate_backend_binding(plan)
  assert plan['launch_permitted'] is True
  # Exact existing model/lineage files, same as the working dispatch; no load or training.
  activation_path=(root/plan['activation']['path']).resolve();activation_path.relative_to(home)
  assert sha(activation_path)==plan['activation']['sha256'];activation=read(activation_path)
  verified=[]
  for part,mapping in [('model',activation['model_file_sha256']),('adapter',{k:v['sha256'] for k,v in activation['files'].items()})]:
   reader=Path(plan[part+'_root']).resolve();reader.relative_to(home)
   for name,digest in mapping.items():
    path=(reader/name).resolve();path.relative_to(reader);assert path.is_file() and sha(path)==digest,(part,name)
    verified.append({'name':part+'/'+name,'sha256':digest,'bytes':path.stat().st_size})
  rec['model_lineage_files_verified']=verified;rec['adapter_loaded_by_KIVI']=False;rec['adapter_loaded_by_H']=False;rec['adapter_audited_but_unused']=True
  assert not (root/plan['batch_output']).exists() and all(not (root/x).exists() for x in plan['outputs'].values()),'Existing quality output; no resume'
  queue=held(control,'fresh_squeue',['squeue','--me','--noheader','--format=%i|%T|%j']);rec['pre_submit_queue']=queue;assert queue['actual_exit_code']==0
  existing=[];owned_GPU_requests=0;owned_namespaces=[]
  for line in queue['stdout'].splitlines():
   fields=[x.strip() for x in line.split('|')];job=fields[0];assert re.fullmatch(r'[0-9_]+',job)
   detail=held(control,'existing_job_'+job,['scontrol','--oneliner','show','job',job]);assert detail['actual_exit_code']==0
   existing.append(dict(job_id=job,**detail));text=detail['stdout']
   assert 'Command='+str(here/'batch.sbatch') not in text,'Same output namespace already submitted'
   if 'qcomem' in fields[2].lower() or '/qcomem_align_codex_20260911/' in text:
    assert fields[2]!=plan['job_name'],'Same cell job name already active or pending'
    match=re.search(r'(?:^| )ReqTRES=([^ ]+)',text);assert match,'Cannot establish owned requested resources'
    resources=dict(x.split('=',1) for x in match.group(1).split(','))
    requested=int(resources.get('gres/gpu','0'))
    assert requested==1,'Unexpected owned GPU count; inspect rather than infer permission'
    owned_GPU_requests+=requested
    command_match=re.search(r'(?:^| )Command=([^ ]+)',text);assert command_match
    namespace=str(Path(command_match.group(1)).parent);Path(namespace).resolve().relative_to(home)
    owned_namespaces.append(namespace)
  assert plan['concurrency_policy']['maximum_owned_requested_GPUs']==4 and plan['concurrency_policy']['GPUs_per_job']==1
  assert owned_GPU_requests==0,'KVDirect serial series requires fresh requestedGPU0 and root-owned reservation transition'
  assert owned_GPU_requests+1<=4,'Four requested GPU limit including pending would be exceeded'
  rec['owned_GPU_requests_before_submission']=owned_GPU_requests
  rec['owned_GPU_requests_after_submission']=owned_GPU_requests+1
  rec['existing_job_details']=existing
  processes=held(control,'fresh_owned_processes',['ps','-u','liuhanzuo','-o','pid=,ppid=,args=']);assert processes['actual_exit_code']==0
  workers=[line for line in processes['stdout'].splitlines() if '/qcomem_align_codex_20260911/' in line and any(token in line for token in ('run_batch.py','resource_worker.py','run_quality.py','resource_launch.py'))]
  rec['access_host_owned_workers']=workers
  rec['CPU_loading_check_scope']='Slurm all owned job commands across assigned nodes plus access-host ps; access-host ps is not compute-node process evidence. Allocated worker performs actual GPU admission.'
  assert all(any(namespace+'/' in line for namespace in owned_namespaces) for line in workers),'Unregistered task-owned access-host loading process; inspect before launch'
  # Read-only diagnostics on access host, never treated as allocated-GPU identity or free-memory admission.
  gpu=held(control,'access_host_gpu',['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'])
  apps=held(control,'access_host_apps',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])
  assert gpu['actual_exit_code']==apps['actual_exit_code']==0
  rec['access_host_GPU_diagnostics']={'GPU':gpu,'apps':apps,'not_allocation_identity_or_admission':True}
  shell=(here/'batch.sbatch').read_text();assert all(flag in shell for flag in ('#SBATCH --partition=gpu','#SBATCH --constraint=l20d','#SBATCH --exclude=gpu-node2,gpu-node8','#SBATCH --nodes=1')) and '#SBATCH --nodelist=' not in shell
  assert '#SBATCH --gres=gpu:1' in shell and '#SBATCH --dependency=singleton' in shell and '#SBATCH --no-requeue' in shell
  assert not (root/plan['batch_output']).exists(),'Fresh output check before sbatch'
  argv=commands['sbatch_argv'];rec.update(status='submitting',submission_invoked=True,argv=argv,qualified_nodes=plan['backend_qualification']['qualified_nodes']);save(receipt,rec)
  submitted=held(control,'sbatch',argv,cwd=str(root))
  rec.update(sbatch_pid=submitted['pid'],sbatch_actual_exit_code=submitted['actual_exit_code'],sbatch_actual_parent_wait=submitted['actual_parent_wait'],sbatch_process_exit_observed=submitted['process_exit_observed'],sbatch_stdout=submitted['stdout'],sbatch_stderr=submitted['stderr']);save(receipt,rec)
  assert submitted['actual_exit_code']==0,'Actual sbatch nonzero; never retry'
  job=submitted['stdout'].strip().split(';')[0];assert re.fullmatch('[0-9]+',job),'Ambiguous returned job ID; never resubmit'
  rec.update(status='slurm_submitted',job_id=job,submitted_at=now());save(receipt,rec)
  snap=held(control,'submitted_squeue',['squeue','--jobs='+job,'--noheader','--format=%i|%T|%j|%N|%R']);rec['compact_queue_snapshot']=snap
  rec['device_scope']='Actual allocated identity comes from this worker, not access-host diagnostics or prior four-arm run. Preserve separate acquisition/device timing; no merged speed ranking.'
  rec['finished_at']=now();save(receipt,rec)
  print(json.dumps({'status':'slurm_submitted','job_id':job,'receipt_sha256':sha(receipt),'GPU_running_claimed':False}))
 except BaseException as error:
  if rec['status']!='slurm_submitted':rec.update(status='submission_failed_or_uncertain_no_retry' if rec['submission_invoked'] else 'failed_before_submission',error=repr(error),finished_at=now())
  save(receipt,rec);raise
if __name__=='__main__':main()
