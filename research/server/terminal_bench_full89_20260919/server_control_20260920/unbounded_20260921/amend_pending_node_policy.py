"""Apply the user's any-node policy only to held, never-started Encbank jobs."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

S = Path('/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
U = S / 'unbounded_20260921'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for arm, job in [('k12', '115838'), ('k48', '115839')]:
    h = S / (arm + '_unbounded_20260921')
    state = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    assert 'JobState=PENDING ' in state and 'Reason=JobHeldUser ' in state
    assert 'ReqNodeList=(null)' in state
    assert not (h / 'run_encbank').exists() and not (h / 'owner_registration.json').exists()
    archive = h / 'pre_any_node_20260921'
    archive.mkdir(exist_ok=False)
    manifest = json.loads((h / 'server_source_manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha(h / name) == digest, name
        shutil.copy2(h / name, archive / name)
    for name in ['server_source_manifest.json', 'submission.json', 'server_preflight.json']:
        shutil.copy2(h / name, archive / name)
    (archive / 'held_slurm_state.txt').write_text(state)
    plan = json.loads((h / 'plan.json').read_text())
    before_plan = sha(h / 'plan.json')
    plan['scheduling_node_policy'] = 'any_eligible_gpu_partition_node'
    plan['allocated_host_runtime_preflight'] = True
    # These lists remain evidence of previous qualification, not an admission allowlist.
    plan['prior_container_qualified_nodes'] = plan.pop('container_qualified_nodes')
    plan['prior_container_qualified_hostnames'] = plan.pop('container_qualified_hostnames')
    (h / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    source = (h / 'server_preflight.py').read_text()
    needle = "                assert os.uname().nodename in plan['container_qualified_hostnames'], 'This node has not been qualified for managed Apptainer'"
    assert source.count(needle) == 1
    source = source.replace(needle, "                assert plan['scheduling_node_policy'] == 'any_eligible_gpu_partition_node'")
    needle = '            container_result = record\n'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + "            if plan.get('allocated_host_runtime_preflight'):\n                from runtime_node_preflight import check_runtime_node\n                container_result = dict(record, allocated_host_runtime=check_runtime_node(plan))\n")
    (h / 'server_preflight.py').write_text(source)
    shutil.copy2(U / 'runtime_node_preflight.py', h / 'runtime_node_preflight.py')
    script = (h / 'server.slurm').read_text()
    assert '#SBATCH --nodelist=gpu-node1\n' in script
    (h / 'server.slurm').write_text(script.replace('#SBATCH --nodelist=gpu-node1\n', ''))
    qualification = json.loads((h / 'container_qualification.json').read_text())
    qualification['plan_sha256'] = sha(h / 'plan.json')
    qualification['node_policy_amendment'] = dict(previous_plan_sha256=before_plan,
        task_setup_qualification_unchanged=True, all_task_qualification_hostname='gpu-host',
        allocated_host_smoke_required=True, scientific_settings_unchanged=True)
    for name, digest in qualification['backend_hashes'].items():
        assert sha(h / name) == digest, name
    (h / 'container_qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
    names = sorted(set(manifest) | {'runtime_node_preflight.py'})
    for name in names:
        if name.endswith('.py'):
            compile((h / name).read_text(), str(h / name), 'exec')
    (h / 'server_source_manifest.json').write_text(json.dumps({n: sha(h / n) for n in names}, indent=2) + '\n')
    receipt = dict(status='amended_pending_only', epoch=time.time(), job_id=job,
        root=str(h), reason='User requested any eligible node on shared cluster filesystem',
        original_submission_preserved=True, original_sources=str(archive),
        prior_source_manifest_sha256=sha(archive / 'server_source_manifest.json'),
        current_source_manifest_sha256=sha(h / 'server_source_manifest.json'),
        current_plan_sha256=sha(h / 'plan.json'), original_requested_node='gpu-node1',
        current_requested_node=None, no_model_calls_before_amendment=True,
        held_slurm_state=state, selection_sha256=plan.get('selection_sha256'),
        task_concurrency=plan['concurrent_tasks'], model_and_task_settings_unchanged=True)
    (h / 'node_policy_amendment.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
