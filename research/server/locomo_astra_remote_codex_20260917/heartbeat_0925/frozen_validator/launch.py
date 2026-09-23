"""Serial CPU Codex CLI judge launcher. No retry; preserve completed/failed attempts."""
from pathlib import Path
import argparse, collections, datetime, hashlib, json, os, re, subprocess, sys, traceback
H=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def save(p,x):
 with p.open('x',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def argv_for(plan,batch,out):
 c=plan['codex'];argv=[c['executable'],'exec','-m',plan['model'],'--ephemeral','--ignore-user-config','--skip-git-repo-check','-s','read-only','-C',c['working_directory'],'--json','--color','never','--output-schema',str(H/batch['schema']),'-o',str(out/'final.json')]
 for value in c['config']:argv+=['-c',value]
 for feature in c['disabled_features']:argv+=['--disable',feature]
 return argv+['-']
def validate_final(plan,batch,final):
 assert set(final)=={'protocol','batch_id','judgments'} and final['protocol']==plan['protocol'] and final['batch_id']==batch['batch_id']
 values=final['judgments'];assert type(values) is list and len(values)==batch['n']
 assert len({x['case_id'] for x in values})==len(values) and {x['case_id'] for x in values}==set(batch['case_ids'])
 for x in values:
  assert set(x)=={'case_id','correct','rationale'} and type(x['correct']) is int and x['correct'] in [0,1] and type(x['rationale']) is str and 0<len(x['rationale'])<=240
 return {x['case_id']:x for x in values}
def validate_events(plan,path):
 events=[json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
 assert events and sum(x.get('type')=='turn.completed' for x in events)==1
 assert any(x.get('type')=='thread.started' for x in events)
 assert not any(x.get('type')=='turn.failed' for x in events)
 notices=[]
 def benign_notice(message):
  return message in ["Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`.","Falling back from WebSockets to HTTPS transport. request timed out"] or re.fullmatch(r'Reconnecting\.\.\. [1-5]/5 \(request timed out\)',message) is not None
 for event in events:
  if event.get('type')=='error':
   assert benign_notice(event.get('message','')),event
   notices.append(event)
 identities=[]
 def inspect(obj):
  if isinstance(obj,dict):
   for k,v in obj.items():
    if k in ['model','model_slug','model_name'] and isinstance(v,str):
     identities.append({'field':k,'value':v});assert v==plan['model'],('runtime model differs',v)
    inspect(v)
  elif isinstance(obj,list):
   for v in obj:inspect(v)
 for e in events:
  if e.get('type','').startswith('item.'):
   item=e.get('item',{});kind=item.get('type')
   if kind=='error':
    assert benign_notice(item.get('message','')),item
    notices.append(e)
   else:assert kind in ['reasoning','agent_message'],('tool or unexpected item',kind)
  assert 'model_switch' not in e.get('type','');inspect(e)
 return {'retained_exact_infrastructure_notices':notices,'event_count':len(events),'runtime_identity_fields':identities,'backend_snapshot':None,'thread_ids':[e.get('thread_id') for e in events if e.get('type')=='thread.started'],'usage':[e.get('usage') for e in events if e.get('type')=='turn.completed']}
def completed(plan,batch,out):
 receipt=read(out/'receipt.json');assert receipt['status']=='COMPLETE_VALIDATED' and receipt['actual_exit_code']==0 and receipt['actual_parent_wait'] and receipt['process_exit_observed']
 assert receipt['plan_sha256']==sha(H/'plan.json') and receipt['prompt_sha256']==batch['prompt_sha256']
 if receipt.get('reused_prior_completed_batch'):
  allowed={x['batch_id']:x for x in plan['continuation_revision']['prior_completed_batches']}
  assert batch['batch_id'] in allowed
  binding=allowed[batch['batch_id']];origin=binding['source_receipt']
  assert receipt['source_completed_receipt']==origin and sha(Path(origin['path']))==origin['sha256']
  prior=read(Path(origin['path']))
  assert prior['status']=='COMPLETE_VALIDATED' and prior['actual_exit_code']==0 and prior['actual_parent_wait'] and prior['process_exit_observed']
  assert prior['plan_sha256']==binding['source_plan_sha256']
  assert receipt['actual_argv']==prior['actual_argv'] and receipt['output_sha256']==prior['output_sha256']==binding['output_sha256']
  assert receipt['prompt_sha256']==prior['prompt_sha256']==batch['prompt_sha256']
  assert receipt['schema_sha256']==prior['schema_sha256']==batch['schema_sha256']
  assert all(sha(Path(origin['path']).parent/n)==d for n,d in binding['output_sha256'].items())
 elif receipt.get('reused_original_first_batch'):
  assert batch['batch_id']=='b0000'
  origin=receipt['original_saved_failure'];assert sha(Path(origin['path']))==origin['sha256']
  old=read(Path(origin['path']));assert old['actual_exit_code']==0 and old['actual_parent_wait'] and old['process_exit_observed']
  assert receipt['actual_argv']==old['actual_argv']
 else:assert receipt['actual_argv']==argv_for(plan,batch,out)
 assert all(sha(out/n)==h for n,h in receipt['output_sha256'].items())
 validate_events(plan,out/'events.jsonl');return validate_final(plan,batch,read(out/'final.json'))
def reused_judgments(plan):
 binding=plan['reused_judgments_file'];assert sha(H/binding['path'])==binding['sha256']
 data=read(H/binding['path']);assert data['protocol']==plan['protocol'] and data['exact_context_only']
 values=data['judgments'];assert len(values)==plan['reused_judgments'] and set(values)==set(data['origins'])
 for cid,value in values.items():
  assert value['case_id']==cid and type(value['correct']) is int and value['correct'] in [0,1] and 0<len(value['rationale'])<=240
  origin=data['origins'][cid];assert sha(Path(origin['receipt']))==origin['receipt_sha256']
 return values

def aggregate(plan,run):
 judgments=reused_judgments(plan)
 for batch in plan['batches']:
  values=completed(plan,batch,run/batch['batch_id']);assert not set(values)&set(judgments);judgments.update(values)
 assert len(judgments)==plan['unique_judgments']==1540
 mapping=[json.loads(x) for x in (H/'private_method_mapping.jsonl').read_text().splitlines()]
 per_item=[dict(x,**{'correct':judgments[x['case_id']]['correct'],'rationale':judgments[x['case_id']]['rationale']}) for x in mapping]
 means={}
 for arm in plan['arms']:
  rows=[x for x in per_item if x['arm']==arm];assert len(rows)==1540 and len({x['source_id'] for x in rows})==1540
  means[arm]={'n':1540,'correct':sum(x['correct'] for x in rows),'accuracy_percent':100*sum(x['correct'] for x in rows)/1540,'by_category':{str(cat):{'n':sum(x['category']==cat for x in rows),'accuracy_percent':100*sum(x['correct'] for x in rows if x['category']==cat)/sum(x['category']==cat for x in rows)} for cat in [1,2,3,4]}}
 save(run/'astra_semantic_results.json',{'protocol':plan['protocol'],'plan_sha256':sha(H/'plan.json'),'status':'ALL_INCREMENTAL_BATCHES_ACTUAL0_AND_PRIOR_REUSE_VALIDATED','new_judgments':plan['new_judgments'],'reused_judgments':plan['reused_judgments'],'new_batch_count':len(plan['batches']),'model_alias':plan['model'],'reasoning_effort':plan['reasoning_effort'],'unique_judgments':len(judgments),'per_item':per_item,'arms':means,'category5':'separate saved native refusal diagnostic; excluded','original_GPT4o_fullJudge':None,'original_Table1_semantic_value':None,'bootstrap':None,'not_literature_comparable':True})
def main():
 p=argparse.ArgumentParser();p.add_argument('--expected-plan-sha256',required=True);p.add_argument('--max-new-batches',type=int,default=1);a=p.parse_args();assert a.max_new_batches>=0
 assert sha(H/'plan.json')==a.expected_plan_sha256
 plan=read(H/'plan.json');manifest=read(H/'manifest.json');assert all(sha(H/n)==h for n,h in manifest['files'].items())
 c=plan['codex'];work=Path(c['working_directory']).resolve();assert not work.is_relative_to(H.parents[2].resolve()) and work.is_dir() and not list(work.iterdir())
 for parent in [work,*work.parents]:assert not any((parent/n).exists() for n in ['AGENTS.md','AGENTS.override.md','.codex','.git']),str(parent)
 assert sha(Path(c['executable']))==c['executable_sha256'];assert all(sha(Path(n))==h for n,h in plan['frozen_source_bindings'].items())
 run=H/'run';run.mkdir(exist_ok=True);lock=run/'launcher.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  reused_judgments(plan)
  launched=0
  for batch in plan['batches']:
   for field in ['input','prompt','schema']:assert sha(H/batch[field])==batch[field+'_sha256']
   out=run/batch['batch_id']
   if out.exists():completed(plan,batch,out);continue
   if a.max_new_batches and launched>=a.max_new_batches:break
   out.mkdir();argv=argv_for(plan,batch,out);receipt={'status':'STARTING','plan_sha256':a.expected_plan_sha256,'batch_id':batch['batch_id'],'prompt_sha256':batch['prompt_sha256'],'schema_sha256':batch['schema_sha256'],'model_alias':plan['model'],'reasoning_effort':plan['reasoning_effort'],'executable_sha256':c['executable_sha256'],'actual_argv':argv,'started_at':now(),'automatic_retry':False}
   save(out/'start.json',receipt)
   env=os.environ.copy()
   for key in ['CODEX_THREAD_ID','CODEX_INTERNAL_ORIGINATOR_OVERRIDE','CODEX_MANAGED_BY_CODEX','CODEX_EXEC_ORIGINATOR','CODEX_CLI_PATH','OPENAI_API_KEY']:env.pop(key,None)
   try:
    with (out/'events.jsonl').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
     child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,cwd=work,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
     receipt['pid']=child.pid;child.communicate((H/batch['prompt']).read_bytes());receipt.update(actual_exit_code=child.returncode,actual_parent_wait=True,process_exit_observed=True)
    assert child.returncode==0,('actual CLI exit nonzero',child.returncode)
    receipt['events_summary']=validate_events(plan,out/'events.jsonl');validate_final(plan,batch,read(out/'final.json'))
    receipt.update(status='COMPLETE_VALIDATED',finished_at=now(),output_sha256={n:sha(out/n) for n in ['events.jsonl','stderr.log','final.json']});save(out/'receipt.json',receipt)
   except BaseException as error:
    if 'child' in locals() and child.poll() is None:
     child.terminate();child.wait();receipt.update(actual_exit_code=child.returncode,actual_parent_wait=True,process_exit_observed=True)
    receipt.update(status='FAILED_PRESERVED_STOP_NO_RETRY',finished_at=now(),error=repr(error),traceback=traceback.format_exc());save(out/'failure.json',receipt);raise
   launched+=1;print(json.dumps({'batch_id':batch['batch_id'],'status':'COMPLETE_VALIDATED','new_batches_this_invocation':launched}),flush=True)
  if all((run/b['batch_id']/'receipt.json').exists() for b in plan['batches']) and not (run/'astra_semantic_results.json').exists():aggregate(plan,run)
 finally:
  assert lock.read_text()==str(os.getpid());lock.unlink()
if __name__=='__main__':main()
