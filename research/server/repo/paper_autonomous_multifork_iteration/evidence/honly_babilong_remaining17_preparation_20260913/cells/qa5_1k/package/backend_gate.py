"""No framework imports. Final qualification binding is mandatory before launch."""
import csv,io
from pathlib import PurePosixPath
from protocol import local, read, sha

RUNTIME={'torch':'2.10.0+cu128','transformers':'5.5.4','peft':'0.20.0','triton':'3.6.0'}

def checked_file(spec):
    assert isinstance(spec,dict) and set(spec)=={'path','sha256'}
    path=PurePosixPath(spec['path'])
    assert not path.is_absolute() and '..' not in path.parts
    assert len(spec['sha256'])==64 and sha(local(spec['path']))==spec['sha256']
    return local(spec['path'])

def validate_backend_binding(plan, *, allow_unbound=False):
    backend=plan['kivi_backend'];q=plan.get('backend_qualification')
    if q is None or backend.get('binary') is None:
        assert plan.get('launch_permitted') is False,'Unbound backend cannot permit launch'
        assert allow_unbound,'Remote backend is unbound; formal launch prohibited'
        return False
    assert plan.get('launch_permitted') is True
    assert isinstance(q['compute_capability'],list) and len(q['compute_capability'])==2
    assert all(type(x) is int and 0<=x<=99 for x in q['compute_capability'])
    assert q['runtime_versions']==RUNTIME
    assert q['qualified_nodes'] and all(isinstance(n,str) and n.replace('-','').isalnum() for n in q['qualified_nodes'])
    assert set(q['node_hostnames'])==set(q['qualified_nodes'])
    assert all(hosts and len(hosts)==len(set(hosts)) and all(isinstance(h,str) and h for h in hosts) for hosts in q['node_hostnames'].values())
    checked_file(backend['binary'])
    registration=read(checked_file(q['registration']))
    assert registration['status']=='completed_kernel_qualification_independently_verified'
    assert registration['binary']==backend['binary']
    assert registration['compute_capability']==q['compute_capability']
    assert registration['runtime_versions']==RUNTIME
    assert registration['kernel_source_sha256']==backend['source_sha256']
    assert registration['numerical_thresholds']=={'max_abs':0.01,'rms':0.001}
    assert registration['numeric_cases']==4 and registration['tail_cases']==2 and registration['closed_phases']==7
    assert registration['numeric_checks_passed'] and registration['entry_invariance_and_release_passed']
    assert registration['actual_exit_codes']=={'worker':0,'parent':0,'shell':0,'slurm':0}
    for node in q['qualified_nodes']:assert node in registration['verified_compatible_nodes']
    assert q['node_hostnames']==registration['verified_node_hostnames']
    proof=registration['proof_files']
    assert proof and {'worker','parent','shell','slurm','batch','independent_report'}<=set(proof)
    for spec in proof.values():checked_file(spec)
    # Bind Slurm's node name to the *same run's* actual hostname and GPU UUID.
    batch=read(local(proof['batch']['path']));worker=read(local(proof['worker']['path']))
    rows=list(csv.DictReader(io.StringIO(local(proof['slurm']['path']).read_text(encoding='utf-8')),delimiter='|'))
    jobs=[r for r in rows if r.get('JobID',r.get('JobIDRaw'))==str(registration['job_id'])]
    assert len(jobs)==1 and jobs[0]['State']=='COMPLETED' and jobs[0]['ExitCode']=='0:0'
    node=jobs[0]['NodeList'];assert q['qualified_nodes']==[node]
    assert str(batch['slurm_job_id'])==str(registration['job_id'])
    assert q['node_hostnames']=={node:[batch['node']]}
    assert batch['physical_gpu_identity']['uuid']==worker['physical_gpu_identity']['uuid']==registration['actual_NVML_identity']['uuid']
    report=read(local(proof['independent_report']['path']))
    field=registration['independent_report_status_field']
    assert field in ('status','verdict')
    assert report[field]==registration['independent_report_status']
    assert isinstance(report[field],str) and report[field].startswith('PASS')
    return True

def qualified_node_identity(plan,environ,hostname):
    """Scheduler node names and socket hostnames are distinct verified identities."""
    q=plan['backend_qualification']
    node=environ.get('SLURMD_NODENAME') or environ.get('SLURM_JOB_NODELIST')
    assert node in q['qualified_nodes'],'Unqualified actual Slurm node'
    assert hostname in q['node_hostnames'][node],'Hostname not mapped to qualified Slurm node'
    return {'slurm_node':node,'hostname':hostname}
