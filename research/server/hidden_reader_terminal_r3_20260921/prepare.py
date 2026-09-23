"""Server-only staging of immutable code and task metadata; never reads verifier contents."""
import json,shutil,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
assert str(R).startswith('/srv/encbank/') and not (R/'plan.json').exists()
BASE=Path('/srv/encbank/qcomem_align_codex_20260911')
SRC=BASE/'terminal_bench_full89_20260919/server_control_20260920/k12_unbounded_20260921'
old=json.loads((SRC/'plan.json').read_text())
tasks=['cancel-async-tasks','custom-memory-heap-crash','large-scale-text-editing','log-summary-date-ranges','query-optimize','sqlite-db-truncate']
for name in ['apptainer_environment.py','apptainer_service.py','apptainer_executor.py','cpu_slots.py','host_admission.py','harbor_unbounded.py']:
    shutil.copy2(SRC/name,R/name)
# host_admission only needs atomic save, not the old HTTP transport.
(R/'server_transport.py').write_text('from common import save\n')
previous=BASE/'hidden_reader_jointband_20260921'
shutil.copytree(previous/'vendor',R/'vendor',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(previous/'band_reader.py',R/'band_reader.py')
manifest={}
for task in tasks:
    for path in sorted((Path(old['task_root'])/task).rglob('*')):
        if path.is_file():manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
plan=dict(experiment='Qwen3-8B joint-attention depth in real Terminal-Bench 2.1',
    benchmark_revision=old['revision'],tasks=tasks,arms=['native','n24'],seed=20260921,
    task_selection='Six named 2GiB tasks covering asynchronous code, memory debugging, text editing, logs, SQL optimization, database recovery; chosen before any 8B results.',
    task_root=old['task_root'],host_admission_root=old['host_admission_root'],host_memory_budget_mb=old['host_memory_budget_mb'],
    harbor_python=old['harbor_python'],engine_python='/srv/encbank/Paper_Evolve/.venv/bin/python',
    model='/srv/encbank/comem_sparse_slurm_20260912/models/Qwen3-8B',
    adapter='/srv/encbank/comem_infra_recheck_20260912/adapter',
    adapter_sha256='1deb86bdc89206ab029ca67403fb3f96dda29fc68223eebec4fc49e97ec0eb13',
    allocator_gib=96,context_tokens=40960,hidden_limit_gib=8,chunk_tokens=512,recent_tokens=512,topk=12,hop=4,
    max_new_tokens=None,task_timeout=None,request_timeout=None,max_turns=None,thinking=False,temperature=0,
    policy='No summarization/cropping/artificial generation limit; native position and physical memory boundaries recorded as incomplete.',
    retrieval='iterative token BM25, positive matches then recent fill to min(12,N); sorted chronologically',
    query='First instruction pinned in exact query KV, plus latest >=512 and <1024 tokens; retrieval once per model call; generation keeps growing KV without chunk-boundary retrieval.',
    concurrency='One GPU per task pair; native and n24 sequential in fresh containers, order alternates across six tasks; six pairs may run concurrently.',
    controlled_replay='After both live trials, choose at most three native requests with most archived chunks, tie by step; replay identical histories with native/n36/n24 and the first up to128 recorded output tokens. Two measurements per arm; excluded from live time.',
    quality_scope='Exploratory six-task A/B, not full89 or the separate27B continuation.',
    environment_source=str(SRC),backend_sha256={n:hashlib.sha256((SRC/n).read_bytes()).hexdigest() for n in ['apptainer_environment.py','apptainer_service.py','apptainer_executor.py','cpu_slots.py','harbor_unbounded.py']},
    resource_inventory=[r for r in old['resource_inventory'] if r['task'] in tasks])
(R/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(R/'task_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
env=json.loads((SRC/'comem_harbor_template.json').read_text())['environment']
(R/'environment_template.json').write_text(json.dumps(env,indent=2)+'\n')
for name in ['jobs','results','qualification','mailbox','logs','tmp','cache','pairs','configs']:(R/name).mkdir()
for task in tasks:(R/'pairs'/task).mkdir();(R/'mailbox'/task).mkdir()
print(json.dumps(dict(tasks=tasks,task_files=len(manifest),server_only=True)))
