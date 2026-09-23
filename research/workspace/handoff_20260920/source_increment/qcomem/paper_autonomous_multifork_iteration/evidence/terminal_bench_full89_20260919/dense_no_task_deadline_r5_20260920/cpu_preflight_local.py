"""Frozen protocol, missing-task partition, public input hashes and resource bounds."""
from pathlib import Path
import ast,datetime,hashlib,json,subprocess,sys
H=Path(__file__).resolve().parent;S=H.parent/'dense_parallel_b'
P=json.loads((H/'plan.json').read_text());old=json.loads((S/'plan.json').read_text())
keys=['model_revision','dtype','lora','engine','engine_version','context_tokens','max_new_tokens',
      'temperature','top_p','top_k','seed','reasoning_effort','preserve_thinking','summarization','context_cropping',
      'runtime_offload','automatic_scientific_retries']
assert all(P[k]==old[k] for k in keys)
identity=json.loads((H/'model_recovery_manifest.json').read_text())
assert identity['identity_matches_prior'] and P['model']==identity['model'] and P['model_revision']==identity['revision']
part=json.loads((H/'continuation_partition.json').read_text())
assert set(P['tasks'])==set(part['remaining'])
assert not set(P['tasks']).intersection(part['frozen_never_rerun'])
assert len(P['tasks'])+len(part['frozen_never_rerun'])+len(part['blocked_environment'])+len(part['pending_adjudication'])==89
assert P['task_walltime_seconds'] is None and P['official_task_timeouts'] is False
assert (P['device_cap_gib'],P['kv_pool_gib'],P['max_num_seqs'],P['request_token_budget'])==(240,160,32,2**21)
assert not (H/'submission.json').exists()
for path in H.glob('*.py'):ast.parse(path.read_text())
r=subprocess.run([sys.executable,str(H/'test_capacity_admission.py')],capture_output=True,text=True)
assert r.returncode==0,r.stderr
root=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919').resolve()
manifest=json.loads((H/'task_manifest.json').read_text());checked=[]
for row in manifest['files']:
    name=Path(row['path']);task=name.parts[1]
    if task not in P['tasks'] or row.get('opaque_verifier'):continue
    path=(root/name).resolve();path.relative_to(root)
    assert path.stat().st_size==row['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],path
    checked.append(row['path'])
assert {Path(n).parts[1] for n in checked}==set(P['tasks'])
out=dict(status='PASS',at=datetime.datetime.now().astimezone().isoformat(),GPU_calls=0,
    protocol_fields_identical=keys,remaining_tasks=len(P['tasks']),frozen_tasks=len(part['frozen_never_rerun']),
    public_input_files_checked=len(checked),opaque_verifier_contents_inspected=False,
    capacity_tests=dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),
    plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),
    prerequisite='Use separately registered unchanged API proof; fresh allocated-node resource/model/transport admission remains mandatory.')
(H/'cpu_preflight_local.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')
print(json.dumps(out))
