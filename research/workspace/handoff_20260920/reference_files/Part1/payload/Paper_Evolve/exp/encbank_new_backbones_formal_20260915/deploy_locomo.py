"""Deploy only implementation/config files; atomically replace each remote file."""
import json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
remote='/srv/encbank/encbank_new_backbones_formal_20260915'
assert json.loads((root/'locomo_scope_checks.json').read_text())['passed']
names=['evaluation_scope.py','evaluate.py','verify.py','locomo_scope.json','locomo.slurm',
       'submit_locomo.py','status_remote.py','locomo_scope_checks.json']
archive=root/'locomo_resume.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for name in names:tf.add(root/name,arcname=name)
subprocess.run(['scp',str(archive),'gpu-node1:'+remote+'/locomo_resume.tar.gz'],check=True)
script=r'''
import ast,datetime,json,subprocess,tarfile
from pathlib import Path
root=Path('/srv/encbank/encbank_new_backbones_formal_20260915')
before=json.loads((root/'evaluation_scope.json').read_text())
assert before['scope_id']=='four-benchmarks-locomo-deferred-20260915'
with tarfile.open(root/'locomo_resume.tar.gz') as tf:
    for member in tf.getmembers():
        p=(root/member.name).resolve()
        assert p.is_relative_to(root) and member.isfile() and p.parent==root
        data=tf.extractfile(member).read()
        if p.suffix=='.py':ast.parse(data.decode())
        temp=p.with_name(p.name+'.deploytmp');temp.write_bytes(data);temp.replace(p)
assert json.loads((root/'evaluation_scope.json').read_text())==before
subprocess.run(['scontrol','update','JobId=45014','ArrayTaskThrottle=4'],check=True)
launch=json.loads((root/'launch.json').read_text());launch['maximum_concurrent_gpus']=4
(root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
receipt=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),four_benchmark_scope_unchanged=True,
    maximum_concurrent_gpus=4,locomo_scope='locomo-resumed-astra-20260916',training_unchanged=True)
(root/'locomo_deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''
subprocess.run(['ssh','gpu-node1','/srv/encbank/Paper_Evolve/.venv/bin/python -'],input=script,text=True,check=True)
subprocess.run(['ssh','gpu-node1',f'/srv/encbank/Paper_Evolve/.venv/bin/python -B {remote}/submit_locomo.py'],check=True)
