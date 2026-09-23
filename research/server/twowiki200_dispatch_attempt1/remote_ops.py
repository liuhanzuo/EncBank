"""Remote stdlib-only validation/promotion, sent over SSH stdin; never retries sbatch."""
import argparse,ast,csv,datetime,hashlib,io,json,os,re,subprocess,sys,tarfile,traceback
from pathlib import Path
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def check_files(manifest,root):
 for x in manifest['files']:
  path=(root/x['repo_relative_path']).resolve();path.relative_to(root)
  assert str(path)==x['remote_path'] and path.stat().st_size==x['bytes'] and sha(path)==x['sha256'],x['repo_relative_path']
 return len(manifest['files'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['validate','submit'],required=True);p.add_argument('--control',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--archive-sha256',required=True);p.add_argument('--expected-plan-sha256',required=True);a=p.parse_args()
 home=Path('/srv/encbank').resolve()
 control=Path(a.control).resolve();control.relative_to(home);assert control.is_dir()
 assert sha(control/'upload_manifest.json')==a.manifest_sha256
 manifest=read(control/'upload_manifest.json');root=Path(manifest['remote_root']).resolve()
 assert root==Path('/srv/encbank/qencbank_align_codex_20260911/repo') and len(manifest['files'])==25
 relative='paper_autonomous_multifork_iteration/evidence/encbank_honly_formal_20260911/next_longbench_qa6/twowiki_full200_candidate/package'
 here=root/relative;here.resolve().relative_to(home);root.relative_to(home)
 cache=(home/'qencbank_align_codex_20260911/task_cache/twowiki200_attempt1').resolve();cache.relative_to(home)
 python='/srv/encbank/qencbank_runtime_20260911/python312/bin/python'
 if a.mode=='validate':
  assert not (control/'package_validation.json').exists(),'Refuse repeated package validation'
  assert sha(control/'sparse_package.tar')==a.archive_sha256
  root.mkdir(parents=True,exist_ok=True)
  expected={x['repo_relative_path']:x for x in manifest['files']}
  with tarfile.open(control/'sparse_package.tar','r:') as tar:
   members=tar.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
   for member in members:
    assert member.isfile() and not member.issym() and not member.islnk()
    dest=(root/member.name).resolve();dest.relative_to(root)
    data=tar.extractfile(member).read();assert len(data)==expected[member.name]['bytes'] and hashlib.sha256(data).hexdigest()==expected[member.name]['sha256']
    if dest.exists():assert sha(dest)==expected[member.name]['sha256'],'Existing unequal file; no overwrite'
    else:
     dest.parent.mkdir(parents=True,exist_ok=True)
     with dest.open('xb') as f:f.write(data)
  report={'status':'running_cpu_file_validation','started_at':now(),'files_verified':check_files(manifest,root),'children':[],'GPU_or_Slurm_actions':0}
  save(control/'package_validation.json',report)
  try:
   assert sha(here/'plan.json')==a.expected_plan_sha256
   plan=read(here/'plan.json');assert not (root/plan['batch_output']).exists()
   for rel in [plan['batch_output'],*plan['outputs'].values(),*plan['source_sha256']]: (root/rel).resolve().relative_to(home)
   for key in ('model_root','adapter_root'): Path(plan[key]).resolve().relative_to(home)
   pyfiles=[root/x['repo_relative_path'] for x in manifest['files'] if x['repo_relative_path'].endswith('.py')]
   for path in pyfiles:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
   report['AST_files']=len(pyfiles)
   for leaf in ('tmp','xdg','hf','torch','triton'):
    directory=(cache/leaf).resolve();directory.relative_to(home);directory.mkdir(parents=True,exist_ok=True)
   env=dict(os.environ);env.update(TMPDIR=str(cache/'tmp'),XDG_CACHE_HOME=str(cache/'xdg'),HF_HOME=str(cache/'hf'),TORCH_HOME=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',USE_TORCH='0',USE_TF='0',USE_FLAX='0',PYTHONUTF8='1',PYTHONHASHSEED='42',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
   for arm in ['dense','h16','h8','h4']:
    argv=[python,'-B',str(here/'resource_worker.py'),'--plan',str(here/'plan.json'),'--expected-plan-sha256',a.expected_plan_sha256,'--arm',arm,'--output',str(root/plan['outputs'][arm]),'--check-only']
    with (control/f'check_{arm}.stdout').open('xb') as out,(control/f'check_{arm}.stderr').open('xb') as err:
     child=subprocess.Popen(argv,stdout=out,stderr=err,env=env);code=child.wait()
    report['children'].append({'arm':arm,'argv':argv,'pid':child.pid,'actual_exit_code':code,'actual_parent_wait':True,'process_exit_observed':True});save(control/'package_validation.json',report)
    assert code==0,('CPU check-only failed',arm,code)
   shell=subprocess.run(['bash','-n',str(here/'batch.sbatch')],capture_output=True,text=True,env=env)
   report['shell_syntax']={'argv':shell.args,'actual_exit_code':shell.returncode,'actual_parent_wait':True,'stdout':shell.stdout,'stderr':shell.stderr};assert shell.returncode==0
   report['resolved_storage_root']=str(home)
   report['read_only_existing_interpreter']={'requested':python,'resolved':str(Path(python).resolve()),'system_interpreter_is_not_task_write':True}
   report.update(status='PASS_sparse_files_AST_and_four_CPU_check_only',finished_at=now())
  except BaseException as error:
   report.update(status='failed_no_retry',finished_at=now(),error=repr(error));raise
  finally:save(control/'package_validation.json',report)
  print(json.dumps({'status':report['status'],'report_sha256':sha(control/'package_validation.json'),'files_verified':report['files_verified']}));return
 # An atomic lock is retained on every outcome; an uncertain SSH response never permits resubmission.
 lock=control/'submission.lock';lock.mkdir(exist_ok=False)
 rec={'status':'validating','started_at':now(),'submission_invoked':False,'automatic_retry':False,'plan_sha256':a.expected_plan_sha256}
 receipt=control/'submit_receipt.json';save(receipt,rec)
 try:
  assert read(control/'package_validation.json')['status']=='PASS_sparse_files_AST_and_four_CPU_check_only'
  assert sha(here/'plan.json')==a.expected_plan_sha256;plan=read(here/'plan.json')
  rec['files_verified']=check_files(manifest,root)
  model_manifest=(root/plan['model_expected_manifest']['path']).resolve();model_manifest.relative_to(home)
  assert sha(model_manifest)==plan['model_expected_manifest']['sha256']
  expected=read(model_manifest);reader=Path(expected['remote_destination']).resolve();reader.relative_to(home)
  model_verified=[]
  for part in ('model','adapter'):
   for name,digest in expected[part].items():
    f=(reader/part/name).resolve();f.relative_to(reader/part);assert f.is_file() and sha(f)==digest,(part,name)
    model_verified.append({'name':part+'/'+name,'bytes':f.stat().st_size,'sha256':digest})
  rec['model_files_verified']=model_verified
  assert not (root/plan['batch_output']).exists(),'Output namespace already exists'
  for value in plan['outputs'].values():assert not (root/value).exists()
  queue=subprocess.run(['squeue','--me','--name=qencbank-align-codex','--noheader','--format=%i|%T|%j'],capture_output=True,text=True)
  rec['pre_submit_queue']={'argv':queue.args,'actual_exit_code':queue.returncode,'stdout':queue.stdout,'stderr':queue.stderr};assert queue.returncode==0
  for line in queue.stdout.splitlines():
   job=line.split('|')[0].strip();detail=subprocess.run(['scontrol','--oneliner','show','job',job],capture_output=True,text=True)
   assert detail.returncode==0
   rec.setdefault('existing_job_details',[]).append({'job_id':job,'stdout':detail.stdout,'actual_exit_code':detail.returncode})
   assert 'Command='+str(here/'batch.sbatch') not in detail.stdout,'Same namespace already submitted'
  assert not queue.stdout.strip(),'An active qencbank batch exists; do not submit a duplicate/overlapping namespace'
  processes=subprocess.run(['ps','-u','liuhanzuo','-o','pid=,ppid=,args='],capture_output=True,text=True);assert processes.returncode==0
  owned=[line for line in processes.stdout.splitlines() if '/qencbank_align_codex_20260911/' in line and any(name in line for name in ('run_batch.py','resource_worker.py','run_quality.py'))]
  rec['active_owned_workers']=owned;assert not owned,'An owned model or batch worker is still active'
  gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True);assert gpu.returncode==0
  apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True);assert apps.returncode==0
  busy={r[0].strip() for r in csv.reader(io.StringIO(apps.stdout)) if len(r)>=4}
  available=[r for r in csv.reader(io.StringIO(gpu.stdout)) if len(r)==4 and r[0].strip() not in busy and int(r[3].strip())*2**20>=plan['resource_policy']['minimum_free_bytes']]
  rec['pre_submit_resource_observation']={'GPU_argv':gpu.args,'GPU_actual_exit_code':gpu.returncode,'GPU_stdout':gpu.stdout,'apps_argv':apps.args,'apps_actual_exit_code':apps.returncode,'apps_stdout':apps.stdout,'available_idle_GPU_count':len(available),'not_allocation_identity':True}
  rec['queued_without_idle_GPU_permitted']=True  # Slurm allocates later; run_batch's original stable/free/foreign guards remain unchanged
  argv=['sbatch','--parsable','--nodelist=gpu-node1',str(here/'batch.sbatch')]
  rec.update(status='submitting',submission_invoked=True,argv=argv);save(receipt,rec)
  with (control/'sbatch.stdout').open('xb') as out,(control/'sbatch.stderr').open('xb') as err:
   child=subprocess.Popen(argv,stdout=out,stderr=err,cwd=str(root));rec['sbatch_pid']=child.pid;save(receipt,rec);code=child.wait()
  stdout=(control/'sbatch.stdout').read_text();stderr=(control/'sbatch.stderr').read_text()
  rec.update(sbatch_actual_exit_code=code,sbatch_actual_parent_wait=True,sbatch_process_exit_observed=True,sbatch_stdout=stdout,sbatch_stderr=stderr)
  save(receipt,rec);assert code==0,('sbatch actual nonzero; no retry',code)
  job=stdout.strip().split(';')[0];assert re.fullmatch(r'[0-9]+',job),'Ambiguous job ID; do not resubmit'
  rec.update(status='slurm_submitted',job_id=job,submitted_at=now());save(receipt,rec)
  snap=subprocess.run(['squeue','--jobs='+job,'--noheader','--format=%i|%T|%j|%N|%R'],capture_output=True,text=True)
  rec['compact_queue_snapshot']={'argv':snap.args,'actual_exit_code':snap.returncode,'stdout':snap.stdout,'stderr':snap.stderr}
  rec['finished_at']=now();save(receipt,rec)
  print(json.dumps({'status':'slurm_submitted','job_id':job,'receipt_sha256':sha(receipt),'GPU_running_claimed':False}))
 except BaseException as error:
  if rec['status']!='slurm_submitted':rec.update(status='submission_failed_or_uncertain_no_retry' if rec['submission_invoked'] else 'failed_before_submission',error=repr(error),finished_at=now())
  save(receipt,rec);raise
if __name__=='__main__':main()
