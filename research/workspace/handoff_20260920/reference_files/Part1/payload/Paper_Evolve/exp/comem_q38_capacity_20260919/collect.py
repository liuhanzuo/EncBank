import json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;S=json.loads((ROOT/'submission.json').read_text())
code="import tarfile;from pathlib import Path;r=Path("+repr(S['stage'])+");t=tarfile.open(str(r/'delivery.tar.gz'),'w:gz');[(t.add(str(p),arcname=p.name)) for p in r.iterdir() if p.name in ['results','complete.json','environment.json','parent_exit.json','stdout.log','stderr.log','plan.json','failure.json']];t.close();print(str(r/'delivery.tar.gz'))"
r=subprocess.run(['ssh','-o','BatchMode=yes','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(code)],capture_output=True,encoding='utf8',timeout=45);assert r.returncode==0,r.stderr
subprocess.run(['scp','-q','-o','BatchMode=yes','gpu-node1:'+r.stdout.strip(),str(ROOT/'delivery.tar.gz')],check=True,timeout=60)
target=ROOT/'collected';target.mkdir(exist_ok=True)
with tarfile.open(ROOT/'delivery.tar.gz') as t:t.extractall(target,filter='data')
print('collected')
