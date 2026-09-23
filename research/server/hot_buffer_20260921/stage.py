import ast,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent;OLD=B/'hidden_reader_terminal_r5_20260921'/'query-optimize'
assert not (R/'plan.json').exists(),'Already staged'
for name in ['band_reader.py','layout.py','common.py']:shutil.copy2(OLD/name,R/name)
shutil.copytree(OLD/'vendor',R/'vendor',ignore=shutil.ignore_patterns('__pycache__'))
P=json.loads((OLD/'plan.json').read_text());P.update(experiment='hot buffer real trace replay and streaming qualification',
    parent_roots=str(B/'hidden_reader_terminal_r5_20260921'),decode_retrieval_interval=512,
    variants=['cold36','cold24','exact36_c24','hot36_c12','hot36_c24','hot36_c48','hot36_c24_no_promote','hot24_c24'],
    capacities_chunks=[12,24,48],trace_selection='first 6 completed requests per task, native preferred when available; fixed before submission',
    trace_decode_tokens=1,stream_forced_tokens=1056,repeats=2,approximate_semantics_explicit=True)
(R/'plan.json').write_text(json.dumps(P,indent=2)+'\n');traces={}
taskroots=json.loads((B/'hidden_reader_terminal_r5_20260921'/'roots.json').read_text());taskroots['cancel-async-tasks']=str(B/'hidden_reader_terminal_r2_20260921')
for task,path in sorted(taskroots.items()):
    box=Path(path)/'mailbox'/task
    arm='native' if list(box.glob('native_*.response.json')) else 'n24';items=[]
    for q in sorted(box.glob(arm+'_*.request.json')):
        a=q.with_name(q.name.replace('.request.','.response.'))
        if not a.exists():continue
        request=json.loads(q.read_text());response=json.loads(a.read_text())
        assert response['request_sha256']==hashlib.sha256(q.read_bytes()).hexdigest()
        if response.get('status')!='ok' or not response.get('generated_ids'):continue
        items.append(dict(messages=request['messages'],first_output_token=response['generated_ids'][0],
            source_request=str(q),source_request_sha256=response['request_sha256'],source_arm=arm,
            source_step=request['step'],source_logical_tokens=response['logical_prompt_tokens']))
        if len(items)==6:break
    if items:traces[task]=items
assert len(traces)>=3
(R/'traces.json').write_text(json.dumps(traces,ensure_ascii=False,indent=2)+'\n')
(R/'cache').mkdir()
env=os.environ.copy();env.update(PYTHONPATH=str(R)+':'+str(R/'vendor')+':/srv/encbank/encbank_infra_recheck_20260912/deps',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
for p in R.rglob('*.py'):ast.parse(p.read_text())
check=subprocess.run([P['engine_python'],'-B','-c','import hot_memory,stream_session,gpu_experiment; print("imports passed")'],cwd=R,env=env,capture_output=True,text=True)
(R/'import_check.json').write_text(json.dumps(dict(code=check.returncode,stdout=check.stdout,stderr=check.stderr),indent=2)+'\n')
assert check.returncode==0,check.stderr
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md']}
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
assert 'hotbuf-' not in queue
(R/'ownership_before_submit.json').write_text(json.dumps(dict(epoch=time.time(),queue=queue),indent=2)+'\n')
cmd=['sbatch','--parsable','--job-name=hotbuf-qual-replay','--partition=gpu','--time=UNLIMITED','--no-requeue',
    '--cpus-per-task=4','--mem=64G','--gres=gpu:nvidia_l20d:1','--exclude=gpu-node1,gpu-node2,gpu-node5,gpu-node6,gpu-node7',
    '--chdir='+str(R),'--output='+str(R/'slurm-%j.out'),'--error='+str(R/'slurm-%j.err'),
    '--wrap',P['engine_python']+' -B '+str(R/'launch_gpu.py')]
submitted=subprocess.run(cmd,capture_output=True,text=True);assert submitted.returncode==0,submitted.stderr
receipt=dict(job=submitted.stdout.strip().split(';')[0],argv=cmd,traces={k:len(v) for k,v in traces.items()},frozen_files=len(manifest),epoch=time.time())
(R/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
