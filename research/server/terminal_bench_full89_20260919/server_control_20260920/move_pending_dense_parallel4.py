"""Move only the unstarted allocation after another node passes the same checks."""
import hashlib
import json
import pathlib
import subprocess
import tarfile
import time

H=pathlib.Path('/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/dense_parallel4_20260921')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    job=json.loads((H/'submission.json').read_text())['job_id']
    assert job=='114849'
    initial=json.loads((H/'container_qualification.json').read_text())
    extra=json.loads((H/'container_qualification_node3.json').read_text())
    assert initial['status']==extra['status']=='PASS'
    assert initial['tasks']==extra['tasks']
    assert extra['plan_sha256']==sha(H/'plan.json')
    state=subprocess.check_output(['squeue','-j',job,'-h','-o','%T'],text=True).strip()
    assert state=='PENDING',state
    assert not (H/'run_dense').exists() and not (H/'execution/owner_registration.json').exists()
    assert not (H/'pending_node_amendment.json').exists()
    manifest=json.loads((H/'server_source_manifest.json').read_text())
    for name,digest in manifest.items():assert sha(H/name)==digest,name
    before=H/'before_node3_allocation.tar.gz'
    with tarfile.open(before,'x:gz') as tar:
        for name in list(manifest)+['server_source_manifest.json','container_qualification.json','submission.json']:
            tar.add(H/name,arcname=name)
    p=json.loads((H/'plan.json').read_text());before_sha=sha(H/'plan.json')
    p['container_qualified_nodes']=['gpu-node1','gpu-node3']
    p['container_qualified_hostnames']=['gpu-host','gpu-host']
    save(H/'plan.json',p)
    script=(H/'server.slurm').read_text().replace('--nodelist=gpu-node1','--nodelist=gpu-node3')
    (H/'server.slurm').write_text(script)
    supervisor=(H/'server_job.py').read_text()
    marker="    assert os.environ.get('SLURM_JOB_ID') and sys.platform == 'linux'"
    assert supervisor.count(marker)==1
    supervisor=supervisor.replace(marker,marker+"\n    os.environ['XDG_RUNTIME_DIR']='/run/user/'+str(os.getuid())\n    os.environ['DBUS_SESSION_BUS_ADDRESS']='unix:path='+os.environ['XDG_RUNTIME_DIR']+'/bus'")
    (H/'server_job.py').write_text(supervisor)
    extra.update(plan_sha256=sha(H/'plan.json'),tested_plan_sha256=before_sha,
         node_extension_only=['container_qualified_nodes','container_qualified_hostnames'],
         qualified_nodes=p['container_qualified_nodes'],qualified_hostnames=p['container_qualified_hostnames'],
         original_qualification_preserved_in=str(before),additional_node='gpu-node3')
    save(H/'container_qualification.json',extra)
    save(H/'server_source_manifest.json',{name:sha(H/name) for name in manifest})
    record=dict(epoch=time.time(),job_id=job,prior_state=state,worker_not_started=True,
        backup=str(before),backup_sha256=sha(before),new_source_manifest_sha256=sha(H/'server_source_manifest.json'),
        explanation='Original node full; move same unstarted allocation to newly qualified node; no extra GPU job or model restart')
    result=subprocess.run(['scontrol','update','JobId='+job,'ReqNodeList=gpu-node3'],capture_output=True,text=True)
    record.update(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    save(H/'pending_node_amendment.json',record)
    print(json.dumps(record));assert result.returncode==0,result.stderr


if __name__=='__main__':main()
