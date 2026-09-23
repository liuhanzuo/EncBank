import datetime,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
receipt=root/'eval_owner_launch.json'
assert not receipt.exists(),'Already launched; inspect before restart'
assert json.loads((root/'data/quality_spec.json').read_text())['n']==400
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (root/'eval_owner.stdout.log').open('ab') as o,(root/'eval_owner.stderr.log').open('ab') as e:
    p=subprocess.Popen(['/usr/bin/python3','-I','-B','-u',str(root/'eval_owner.py')],cwd=str(root),env=env,
        stdout=o,stderr=e,stdin=subprocess.DEVNULL,start_new_session=True)
r=dict(pid=p.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat());receipt.write_text(json.dumps(r,indent=2));print(json.dumps(r))
