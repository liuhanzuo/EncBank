"""CPU-only extraction of existing Dense evidence. No experiment/service commands."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,datetime,hashlib,json,os,tarfile
ROOT=Path('/srv/encbank/legacy_workspace');B=ROOT/'paper_autonomous_multifork_iteration/evidence/terminal_bench_full89_20260919';H=Path(__file__).resolve().parent;RT=ROOT/'.runtime/terminal_bench_full89_20260919'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
original=B/'final_handoff_20260921/closed_verification.json';dense=load(original)['arms']['dense'];assert len(dense['retained_normal'])==74
save(H/'DENSE_ORIGINAL_REGISTRATION.json',dict(source=ref(original),dense=dense,note='Original completion registration is preserved; this supplement separately audits truncation.'))
selected=defaultdict(list)
for t in dense['retained_normal']:selected[Path(t['path']).relative_to(RT/'results').parts[0]].append(t)
legacy_roots={r for r in selected if 'no_task_deadline' not in r};assert sum(len(selected[r]) for r in legacy_roots)==38
files=set();task_rows=[];calls=[];source_map={};warnings=[]
for root,tasks in sorted(selected.items()):
 ev='toolchain_fix' if root=='dense_vllm_batch2_toolchain_fix' else root;D=B/ev;box=RT/('local_rpc_'+ev)/'dense';plan=load(D/'plan.json');code=(D/'tb_agent_rpc.py').read_text(encoding='utf8')
 no_task_deadline='self.limit=None' in code;timeouts={x['task']:x['agent_seconds'] for x in plan.get('timeout_inventory',[])}
 wanted={x['task']:Path(x['path']).parent.name for x in tasks};requests=defaultdict(list)
 for p in box.glob('*.request.json'):
  q=load(p)
  if q['task'] in wanted and q['trial']==wanted[q['task']]:requests[q['task']].append((p,q))
 source_map[root]=dict(evidence_root=D.relative_to(ROOT).as_posix(),rpc_root=box.relative_to(ROOT).as_posix(),plan=ref(D/'plan.json'),rpc_code=ref(D/'tb_agent_rpc.py'),worker_code=ref(D/'agent_worker.py'),cumulative_task_deadline_disabled=no_task_deadline)
 for n in ['plan.json','tb_agent_rpc.py','agent_worker.py','windows_owner.py','file_bridge.py','science_unchanged.json','recovery_check.json','submission.json','native_launch.json','model_hashes_completion.json']:
  if (D/n).exists():files.add(D/n)
 for n in ['harbor_receipt.json','owner_registration.json','owner_complete.json','controller_failure.json','stop_receipt.json','worker_ready.json','transport_snapshot.json']:
  if (D/'execution'/n).exists():files.add(D/'execution'/n)
 for t in sorted(tasks,key=lambda x:x['task']):
  task=t['task'];result_path=Path(t['path']);assert sha(result_path)==t['sha256'];raw=load(result_path);assert not raw.get('exception_info')
  files.add(result_path)
  for n in ['config.json','trial.log','exception.txt']:
   if (result_path.parent/n).exists():files.add(result_path.parent/n)
  rollouts=(raw.get('agent_result') or {}).get('rollout_details',[]);assert len(rollouts)==1,(task,'Unexpected rollout structure')
  roll=rollouts[0];extra=roll['extra'];count=len(roll['completion_token_ids']);rr=sorted(requests[task],key=lambda p:p[1]['step']);assert len(rr)==count>0,(task,len(rr),count)
  assert [q['step'] for _,q in rr]==list(range(count))
  result_ref=ref(result_path);rows=[]
  for i,(qpath,q) in enumerate(rr):
   rid=q['request_id'];ap=box/(rid+'.response.json');assert ap.exists(),(task,rid,'Missing raw response')
   a=load(ap);assert all(a[k]==q[k] for k in ['request_id','task','task_id','step'])
   assert a['generated_tokens']==len(a['generated_ids']) and a['prompt_tokens']==len(a['prompt_ids'])
   assert a['generated_ids']==roll['completion_token_ids'][i] and a['prompt_ids']==roll['prompt_token_ids'][i],(task,rid,'Result token sequence mismatch')
   for k in ['request_id','task','task_id','step','status','finish_reason','stop_reason','generated_tokens','prompt_tokens']:
    assert k in extra and len(extra[k])==count and extra[k][i]==a.get(k),(task,rid,k)
   finish=a.get('finish_reason');gen=a['generated_tokens'];effective=min(plan['max_new_tokens'],plan['context_tokens']-a['prompt_tokens'])
   token_cap=finish=='length';deadline=a.get('status')=='deadline';cancelled=a.get('status')=='cancelled'
   cut='GENERATION_TOKEN_CAP' if token_cap and gen==plan['max_new_tokens'] else ('CONTEXT_REMAINING_TOKEN_CAP' if token_cap and gen==effective and effective<plan['max_new_tokens'] else ('LENGTH_OTHER_OR_UNKNOWN' if token_cap else 'NONE_REPORTED'))
   r=dict(task=task,trial=q['trial'],root=root,step=i,request_id=rid,raw_status=a.get('status'),finish_reason_present='finish_reason' in a,finish_reason=finish,stop_reason=a.get('stop_reason'),
    generated_tokens=gen,prompt_tokens=a['prompt_tokens'],configured_max_new_tokens=plan['max_new_tokens'],effective_max_new_tokens=effective,token_limit_truncated=token_cap,token_limit_kind=cut,
    time_deadline_truncated=deadline,cancelled=cancelled,cancellation_cause='UNSPECIFIED' if cancelled else None,remaining_seconds_at_client=q.get('remaining_seconds'),client_created_epoch=q.get('client_created_epoch'),
    configured_cumulative_task_limit_seconds=None if no_task_deadline else timeouts.get(task),per_call_watchdog_seconds=plan.get('per_call_transport_watchdog_seconds'),request_seconds=a.get('request_seconds'),ttft_seconds=a.get('ttft_seconds'),decode_seconds=a.get('decode_seconds'),
    request=ref(qpath),response=ref(ap),raw_request_response_in_archive=root in legacy_roots,
    packaged_result_source=dict(**result_ref,rollout_index=0,call_index=i,metadata_path='agent_result.rollout_details[0].extra',token_path='agent_result.rollout_details[0].completion_token_ids'),crosschecked_raw_reply_against_result=True)
   for kind in ['broker.json','error.json','cancel.json']:
    p=box/(rid+'.'+kind)
    if p.exists():
     r[kind]=ref(p)
     if kind=='cancel.json':r['local_cancel_marker']=load(p)
     if kind=='broker.json':
      broker=load(p);r['broker_status']=broker.get('status')
      if broker.get('proof'):assert broker['proof']['sha256']==r['response']['sha256'] and broker['proof']['bytes']==ap.stat().st_size
     files.add(p)
   if root in legacy_roots:
    files.update([qpath,ap])
    for suffix in ['.transfer','.download']:
     p=ap.with_suffix(suffix)
     if p.exists():files.add(p)
   rows.append(r);calls.append(r)
  receipts=[]
  for n in ['receipts','parent_waits','launches','configs']:
   p=D/'execution'/n/(task+'.json')
   if p.exists():files.add(p);receipts.append(ref(p))
  if not any('/receipts/' in x['path'] for x in receipts):warnings.append(dict(task=task,root=root,note='No per-task OS parent receipt: early shared Harbor batch has aggregate receipt; preserve original boundary.'))
  generated=sum(x['generated_tokens'] for x in rows);assert generated==raw['agent_result']['n_output_tokens'],(task,'Output usage mismatch')
  task_rows.append(dict(task=task,root=root,trial=result_path.parent.name,legacy_38=root in legacy_roots,raw_request_response_in_archive=root in legacy_roots,result=result_ref,receipts=receipts,
   original_reward=raw['verifier_result']['rewards']['reward'],original_exception=raw.get('exception_info'),agent_execution=raw.get('agent_execution'),configured_cumulative_task_limit_seconds=None if no_task_deadline else timeouts.get(task),
   calls=count,generated_tokens=generated,finish_reason_counts=dict(Counter(x['finish_reason'] for x in rows)),length_capped_calls=sum(x['token_limit_truncated'] for x in rows),
   time_deadline_calls=sum(x['time_deadline_truncated'] for x in rows),cancelled_calls=sum(x['cancelled'] for x in rows),unknown_finish_calls=sum(x['finish_reason'] not in ['stop','length'] for x in rows),
   truncation_review_needed=any(x['token_limit_truncated'] or x['time_deadline_truncated'] or x['cancelled'] or x['finish_reason'] not in ['stop','length'] for x in rows),
   all_raw_replies_verified=True,all_reply_metadata_reconstructible_from_packaged_result=True))
 print(json.dumps(dict(root=root,tasks=len(tasks),verified=True)),flush=True)
summary=dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS',tasks=len(task_rows),legacy_tasks=38,calls=len(calls),legacy_calls=sum(x['legacy_38']*x['calls'] for x in task_rows),
 finish_reason_counts=dict(Counter(r['finish_reason'] for r in calls)),length_capped_calls=sum(r['token_limit_truncated'] for r in calls),length_capped_tasks=sum(r['length_capped_calls']>0 for r in task_rows),legacy_length_capped_tasks=sum(r['legacy_38'] and r['length_capped_calls']>0 for r in task_rows),
 time_deadline_calls=sum(r['time_deadline_truncated'] for r in calls),cancelled_calls=sum(r['cancelled'] for r in calls),unknown_finish_calls=sum(r['finish_reason'] not in ['stop','length'] for r in calls),
 total_generated_tokens=sum(r['generated_tokens'] for r in calls),new_model_calls=0,new_gpu_jobs=0,experiments_restarted=False,warning_count=len(warnings))
save(H/'DENSE_TASK_INDEX.json',dict(summary=summary,tasks=task_rows,source_map=source_map,provenance_notes=warnings))
save(H/'DENSE_CALL_INDEX.json',dict(summary=summary,calls=calls,interpretation='Time deadline configuration is not evidence of actual truncation. Deadline and cancelled statuses are distinct. Length finish is flagged even when downstream verifier passes. Unknown is not false.'))
save(H/'TRUNCATION_REVIEW_TASKS.json',dict(policy='Evidence-based candidate list only; no experiment launched and no prior reward overwritten.',tasks=[r for r in task_rows if r['truncation_review_needed']]))
save(H/'LEGACY_38_TASKS.json',dict(tasks=[r['task'] for r in task_rows if r['legacy_38']],raw_request_response_calls=summary['legacy_calls']))
save(H/'summary.json',summary)
fields=['task','root','legacy_38','original_reward','calls','generated_tokens','length_capped_calls','time_deadline_calls','cancelled_calls','unknown_finish_calls','configured_cumulative_task_limit_seconds','truncation_review_needed']
with (H/'DENSE_TASK_INDEX.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in task_rows)
fields=['task','root','step','request_id','raw_status','finish_reason','generated_tokens','prompt_tokens','token_limit_truncated','token_limit_kind','time_deadline_truncated','cancelled','remaining_seconds_at_client','configured_cumulative_task_limit_seconds','request_seconds','raw_request_response_in_archive']
with (H/'DENSE_CALL_INDEX.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in calls)
for p in H.iterdir():
 if p.is_file() and p.suffix in ['.py','.json','.csv','.md'] and p.name not in ['FILE_SHA256.json','package_receipt.json']:files.add(p)
archive=H/'DENSE_EVIDENCE_SUPPLEMENT.tar.gz';assert not archive.exists();manifest=[]
with tarfile.open(archive,'w:gz',compresslevel=2) as tf:
 for i,p in enumerate(sorted(files)):
  p.resolve().relative_to(ROOT.resolve());assert not set(p.parts)&{'ko2','matched_v2','ground_truth','artifacts','tests'}
  st=p.stat();r=ref(p);r['archive_path']='repository/'+r['path'];tf.add(p,arcname=r['archive_path'],recursive=False);after=p.stat();assert (st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  manifest.append(r)
  if i and i%1000==0:print(json.dumps(dict(packaged=i,total=len(files))),flush=True)
save(H/'FILE_SHA256.json',dict(count=len(manifest),files=manifest,total_bytes=sum(r['bytes'] for r in manifest)))
lookup={r['archive_path']:r for r in manifest};count=0
with tarfile.open(archive,'r:gz') as tf:
 for m in tf:
  assert m.isfile() and m.name in lookup
  f=tf.extractfile(m);h=hashlib.file_digest(f,'sha256').hexdigest();assert m.size==lookup[m.name]['bytes'] and h==lookup[m.name]['sha256'];count+=1
assert count==len(manifest)
save(H/'package_receipt.json',dict(status='PASS',archive=ref(archive),manifest=ref(H/'FILE_SHA256.json'),verified_files=count,summary=summary,raw_full_legacy_tasks=38,all_74_complete_results_included=True,all_74_call_metadata_crosschecked=True))
print(json.dumps(load(H/'package_receipt.json')),flush=True)
