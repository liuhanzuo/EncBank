import ast,hashlib,json,os,shutil,subprocess,time,sys
from pathlib import Path
SRC=Path(__file__).resolve().parent;revision=sys.argv[1] if len(sys.argv)>1 else 'r1'
assert revision in ['r1','r2','r3','r4','r5','r6']
ROOT=SRC.parent/('tf27b_hot_qualification_'+revision+'_20260921')
OLD=SRC.parent/'terminal_bench_full89_20260919/server_control_20260920/k12_unbounded_20260921'
assert not ROOT.exists(),'Never overwrite a qualification attempt'
ROOT.mkdir()
names=['common.py','native_common.py','hybrid_reader.py','hybrid_hot.py','batch_cache.py',
    'memory_selectors.py','runtime_identity.py','model_setup.py','qualify_gpu.py','launch_qualification.py']
for n in names:shutil.copy2(SRC/n,ROOT/n)
P=json.loads((OLD/'plan.json').read_text())
P.update(experiment='27B Transformers hot-prefix qualification',allocator_gib=248,engine_python='/srv/encbank/Paper_Evolve/.venv/bin/python')
(ROOT/'plan.json').write_text(json.dumps(P,indent=2)+'\n')
traces=json.loads((SRC.parent/'hot_buffer_20260921/traces.json').read_text())
# Deterministic named real task request; no selection by outcome.
trace=traces['log-summary-date-ranges'][0]
(ROOT/'trace_messages.json').write_text(json.dumps(trace['messages'],indent=2)+'\n')
(ROOT/'trace_source.json').write_text(json.dumps({k:v for k,v in trace.items() if k!='messages'},indent=2)+'\n')
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
for p in ROOT.glob('*.py'):ast.parse(p.read_text())
res=subprocess.run([P['engine_python'],'-B','-c','import hybrid_hot,model_setup,qualify_gpu'],env=env,cwd=ROOT,capture_output=True,text=True)
(ROOT/'import_check.json').write_text(json.dumps(dict(exit_code=res.returncode,stdout=res.stdout,stderr=res.stderr),indent=2)+'\n')
assert res.returncode==0,res.stderr
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file()}
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
assert 'tf27b-' not in queue
(ROOT/'ownership_before_submit.json').write_text(json.dumps(dict(queue=queue,epoch=time.time()),indent=2)+'\n')
cmd=['sbatch','--parsable','--job-name=tf27b-qual-'+revision,'--partition=gpu','--time=UNLIMITED','--no-requeue',
    '--cpus-per-task=8','--mem=80G','--gres=gpu:nvidia_l20d:1','--chdir='+str(ROOT),
    '--output='+str(ROOT/'slurm-%j.out'),'--error='+str(ROOT/'slurm-%j.err'),
    '--exclude=gpu-node6', '--wrap',P['engine_python']+' -B '+str(ROOT/'launch_qualification.py')]
res=subprocess.run(cmd,capture_output=True,text=True);assert res.returncode==0,res.stderr
receipt=dict(job=res.stdout.strip().split(';')[0],root=str(ROOT),argv=cmd,epoch=time.time())
(ROOT/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
