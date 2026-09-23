"""New user-authorized full89 Encbank deployment retest; preserve all predecessors."""
from pathlib import Path
import ast,datetime,hashlib,json
H=Path(__file__).resolve().parent;S=H.parent/'encbank_batch8_service';C=H.parent/'encbank_batch8_refill_preparation'
assert not (H/'plan.json').exists()
def save(n,x):(H/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def put(n,x):(H/n).write_text(x,encoding='utf8',newline='\n')
names=['common.py','batch_cache.py','hybrid_reader.py','memory_selectors.py','session.py','agent_worker.py','holder.py','file_bridge.py','persistence.py','storage_preflight.py','cpu_preflight_remote.py','encbank.slurm','windows_owner.py','tb_agent_rpc.py','host_admission.py','owner_recovery.py','deploy.py','observe.py','sync_queue.py','task_manifest.json','model_hashes_completion.json','ragged_cpu_check.json','encbank_harbor_template.json','.gitattributes','.gitignore']
for n in names:
    value=(S/n).read_text(encoding='utf8').replace('encbank_batch8_service',H.name).replace('qencbank-tb-encbank-batch8-','qencbank-tb-encbank-refill8-')
    put(n,value)
put('fair_share.py','"""No cross-campaign pause; shared host_admission enforces the RAM budget."""\ndef update(needs_capacity,save):\n    return None\n')
put('service_loop.py',(C/'service_loop_refill.py').read_text(encoding='utf8'))
put('batch_cache_refill.py',(C/'batch_cache_refill.py').read_text(encoding='utf8'))
put('cpu_preflight_local.py',(H.parent/'encbank_service/cpu_preflight_local.py').read_text(encoding='utf8'))
P=json.loads((S/'plan.json').read_text());full=json.loads((H.parent/'encbank_service/plan.json').read_text())
P.update(evidence_id='E-ENCBANK-TB21-REFILL8-RETEST-20260919',remote_root=P['remote_root'].replace(S.name,H.name),
    tasks=full['tasks'],resource_inventory=full['resource_inventory'],timeout_inventory=full['timeout_inventory'],
    authorization='2026-09-19 user explicitly requests restore and retest after static-cohort queue starvation. Rerun all89, independent of old scores; retain old deployment outputs.',
    serialization='independent serial Write/prefill; continuous refill ragged decode up to8',refill_interval_seconds=.5,
    deployment_boundary='New scheduler revision, all89 fresh; no pooling or best-of with predecessor15 outcomes. Same official deadlines, k12, model, LoRA and sampling; batch changes may change FP arithmetic.')
assert len(P['tasks'])==89 and len(set(P['tasks']))==89
save('plan.json',P)
cfg=json.loads((H/'encbank_harbor_template.json').read_text());cfg['tasks']=[dict(path='/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919/tasks/'+t) for t in P['tasks']];save('encbank_harbor_template.json',cfg)
d=(H/'deploy.py').read_text().replace("'batch_cache.py','service_loop.py'","'batch_cache.py','batch_cache_refill.py','service_loop.py'")
d=d.replace("assert json.loads((H/'ragged_cpu_check.json').read_text())['status']=='PASS'","assert json.loads((H/'ragged_cpu_check.json').read_text())['status']=='PASS'\n    assert json.loads((H/'service_cpu_check.json').read_text())['status']=='PASS'")
put('deploy.py',d)
manifest=json.loads((H/'task_manifest.json').read_text());R=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919')
for row in manifest['files']:assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
fields=['model','model_revision','dtype','lora','j','adapter_path','adapter_sha256','context_tokens','max_new_tokens','temperature','top_p','top_k','seed','reasoning_effort','preserve_thinking','official_task_timeouts','runtime_offload']
old=json.loads((S/'plan.json').read_text());assert all(P[k]==old[k] for k in fields)
save('task_preparation_receipt.json',dict(status='PASS',public_input_files_checked=len(manifest['files']),tasks=89,protocol_fields_identical=fields,plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),verifier_contents_read=False,model_calls=0))
save('registration.json',dict(at=datetime.datetime.now().astimezone().isoformat(),status='PREPARED_NOT_SUBMITTED',tasks=89,old_results_retained=True,selection_by_old_score=False,storage_boundary='/srv/encbank/',storage_note='Shared BeeGFS fault not claimed repaired; formal storage gate and same-payload bounded persistence remain; no model-generation retry.',prior_cpu_cache_check=str(C/'cpu_attempt2_record.json'),scientific_changes=[],deployment_changes=['Refill free decode slots while old requests are active; protect active session banks from release.']))
for p in H.glob('*.py'):ast.parse(p.read_text())
print(json.dumps(dict(status='PREPARED',tasks=89,path=str(H))))
