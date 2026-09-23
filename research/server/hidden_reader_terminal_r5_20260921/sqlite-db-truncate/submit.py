"""Fresh ownership checks and immutable source records; no action on other jobs."""
import ast,fcntl,json,subprocess,sys,time
from common import ROOT,PLAN as P,save,sha
phase=sys.argv[1];assert phase in ['qualify','run']
lock=(ROOT/'jobs'/'submit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
record=ROOT/'jobs'/('submissions_'+phase+'.json');assert not record.exists(),'Already submitted'
for path in ROOT.rglob('*.py'):ast.parse(path.read_text())
if phase=='qualify':
    manifest={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and (p.suffix in ['.py','.md'] or p.name in ['plan.json','task_manifest.json','environment_template.json'])}
    save(ROOT/'source_manifest.json',manifest)
else:
    from common import verify_sources
    verify_sources()
    from pathlib import Path
    reuse=json.loads((ROOT/'qualification_reuse.json').read_text());origin=Path(reuse['origin'])
    assert json.loads((origin/'jobs'/'qualification_complete.json').read_text())['passed']
    assert json.loads((origin/'jobs'/'qualify_parent_exit.json').read_text())['exit_code']==0
    for name,expected in reuse['backend_hashes'].items():assert sha(ROOT/name)==sha(origin/name)==expected,name
    assert json.loads((ROOT/'dependency_check.json').read_text())['passed']
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
assert not any('bandtb4-' in line for line in queue.splitlines()),'Existing revision4 jobs require review before resubmission'
save(ROOT/'jobs'/('ownership_'+phase+'.json'),dict(epoch=time.time(),queue=queue))
rows=[]
specs=[('qualify',None)] if phase=='qualify' else [('worker',t) for t in P['tasks']]+[('controller',None)]
for kind,task in specs:
    name='bandtb4-'+(task[:28] if task else kind)
    command=['sbatch','--parsable','--job-name='+name,'--partition=gpu','--time=UNLIMITED','--no-requeue',
        '--cpus-per-task='+('4' if kind=='worker' else '8'),'--mem='+('64G' if kind=='worker' else '24G'),
        '--chdir='+str(ROOT),'--output='+str(ROOT/'logs'/(name+'-%j.slurm.out')),'--error='+str(ROOT/'logs'/(name+'-%j.slurm.err'))]
    if kind=='worker':command+=['--gres=gpu:nvidia_l20d:1','--exclude=gpu-node1,gpu-node2,gpu-node5,gpu-node6,gpu-node7']
    else:command+=['--nodelist=gpu-node4']
    command+=['--wrap',P['engine_python']+' -B '+str(ROOT/'launch.py')+' '+kind+(' '+task if task else '')]
    result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0,result.stderr
    row=dict(kind=kind,task=task,job=result.stdout.strip().split(';')[0],epoch=time.time(),argv=command);rows.append(row);save(record,rows)
print(json.dumps(rows,indent=2))
