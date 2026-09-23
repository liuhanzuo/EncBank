"""Partition only delegated, never-started tasks into four immutable GPU runs."""
import hashlib
import json
import shutil
import time
from pathlib import Path

S = Path('/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
U = S / 'unbounded_20260921'
D = U / 'scale4_20260921'

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, v): p.write_text(json.dumps(v, indent=2) + '\n')

transfer = read(D / 'ownership_transfer.json')
selection = read(U / 'selection.json')
registry = dict(version=1, epoch=time.time(), gpu_limit=4, per_gpu_concurrency=8,
                maximum_total_concurrency=32, selection_sha256=sha(U / 'selection.json'),
                transfer_path=str(D / 'ownership_transfer.json'),
                transfer_sha256=sha(D / 'ownership_transfer.json'), runs=[])
dense = S / 'dense_unbounded_20260921'
registry['runs'].append(dict(id='dense', arm='dense', root=str(dense),
                             tasks=read(dense / 'plan.json')['tasks'], role='completed'))
for arm in ['k12', 'k48']:
    old = S / (arm + '_unbounded_20260921')
    record = transfer['arms'][arm]
    p = read(old / 'plan.json')
    assert sha(old / 'plan.json') == record['plan_sha256']
    assert read(old / 'execution/status.json')['state'] == 'draining'
    assert {f.stem for f in (old / 'execution/launches').glob('*.json')} == set(record['launched'])
    registry['runs'].append(dict(id=arm+'_original', arm=arm, root=str(old),
        tasks=record['launched'], role='finish_existing_only'))
    manifest = read(old / 'server_source_manifest.json')
    for name, digest in manifest.items(): assert sha(old / name) == digest, name
    for index, suffix in enumerate(['a', 'b']):
        tasks = record['delegated'][index::2]
        assert tasks and set(tasks).isdisjoint(record['launched'])
        name = arm + '_scale4_' + suffix + '_20260921'
        h = S / name
        h.mkdir(exist_ok=False)
        old_runtime = str(Path(p['rpc_root']).parent)
        new_runtime = str(Path(old_runtime).parent / name)
        ipc = '/srv/encbank/qencbank_runtime_20260911/t89s4' + arm + suffix
        new_job_name = 'qencbank-tb-' + arm + '-scale4-' + suffix
        # Only paths and deployment names are substituted; inference implementation is preserved.
        replacements = [(str(old), str(h)), (old_runtime, new_runtime),
                        (p['ipc_root'], ipc), (p['job_name'], new_job_name)]
        for source_name in manifest:
            data = (old / source_name).read_bytes()
            for before, after in replacements: data = data.replace(before.encode(), after.encode())
            (h / source_name).write_bytes(data)
        plan = read(h / 'plan.json')
        plan.update(tasks=tasks, resource_inventory=[r for r in p['resource_inventory'] if r['task'] in tasks],
            remote_root=str(h), ipc_root=ipc, job_name=new_job_name,
            evidence_id=p['evidence_id'] + '-SCALE4-' + suffix.upper(),
            scale4_parent_root=str(old), scale4_transfer_sha256=sha(D / 'ownership_transfer.json'),
            scale4_partition=suffix, scale4_authorization='User requested more GPUs and higher total concurrency; existing tasks finish normally, delegate only never-started tasks.',
            allocation_policy='At most4 running+pending benchmark GPU allocations; each8 tasks; any eligible node.')
        save(h / 'plan.json', plan)
        template = read(h / 'encbank_harbor_template.json')
        template['job_name'] = name
        template['tasks'] = [dict(path=str(Path(plan['task_root']) / t)) for t in tasks]
        save(h / 'encbank_harbor_template.json', template)
        q = read(old / 'container_qualification.json')
        for source_name, digest in q['backend_hashes'].items(): assert sha(h / source_name) == digest, source_name
        q.update(tasks=tasks, checks=[r for r in q['checks'] if r['task'] in tasks],
                 plan_sha256=sha(h / 'plan.json'),
                 scale4_transfer=dict(source=str(old / 'container_qualification.json'),
                     sha256=sha(old / 'container_qualification.json'), task_and_backend_unchanged=True,
                     allocated_host_smoke_required=True))
        assert len(q['checks']) == len(tasks) and all(r['status']=='PASS' for r in q['checks'])
        save(h / 'container_qualification.json', q)
        save(h / 'selection_provenance.json', dict(selection_path=str(U / 'selection.json'),
            selection_sha256=sha(U / 'selection.json'), transfer_sha256=sha(D / 'ownership_transfer.json'),
            original_root=str(old), tasks=tasks, zero_prior_launches=True, zero_prior_model_calls=True))
        for source_name in manifest:
            if source_name.endswith('.py'): compile((h / source_name).read_text(), str(h / source_name), 'exec')
        save(h / 'server_source_manifest.json', {n:sha(h/n) for n in manifest})
        registry['runs'].append(dict(id=arm+'_'+suffix, arm=arm, root=str(h), tasks=tasks, role='new_shard',
                                    source_manifest_sha256=sha(h/'server_source_manifest.json')))
        print(json.dumps(dict(id=arm+'_'+suffix, tasks=len(tasks), root=str(h))), flush=True)
for arm in ['dense', 'k12', 'k48']:
    tasks = [t for r in registry['runs'] if r['arm']==arm for t in r['tasks']]
    assert len(tasks)==len(set(tasks)) and set(tasks)==set(selection['arms'][arm]['tasks'])
registry['submission_order'] = ['k12_a', 'k48_a', 'k12_b', 'k48_b']
save(D / 'registry.json', registry)
