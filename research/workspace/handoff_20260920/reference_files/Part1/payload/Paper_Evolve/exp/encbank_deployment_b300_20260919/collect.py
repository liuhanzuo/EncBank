import json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CODE=r'''
import json,sys,tarfile
from pathlib import Path
p=json.load(sys.stdin);r=Path(p['stage']);wait=json.loads((r/'parent_exit.json').read_text());assert wait['actual_wait'] and wait['returncode']==0
assert json.loads((r/'complete.json').read_text())['points']==30
dest=r/'results-delivery.tar.gz'
with tarfile.open(str(dest)+'.tmp','w:gz') as tar:
 for path in r.glob('*.json'):tar.add(str(path),arcname=path.name)
 for name in ['results','stdout.log','stderr.log']:tar.add(str(r/name),arcname=name)
Path(str(dest)+'.tmp').replace(dest)
print(json.dumps(dict(path=str(dest),bytes=dest.stat().st_size)))
'''
def main():
    p=json.loads((ROOT/'submission.json').read_text())
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(CODE)],input=json.dumps(p),capture_output=True,encoding='utf-8',timeout=90)
    assert r.returncode==0,r.stderr
    remote=json.loads(r.stdout);dest=ROOT/'delivery.tar.gz';temp=ROOT/'delivery.partial'
    subprocess.run(['scp','-q','-o','BatchMode=yes','gpu-node1:'+remote['path'],str(temp)],check=True,timeout=180);temp.replace(dest)
    output=ROOT/'delivery';output.mkdir(exist_ok=True)
    with tarfile.open(dest) as tar:
        for m in tar.getmembers():assert (output/m.name).resolve().is_relative_to(output.resolve()) and not m.issym() and not m.islnk()
        tar.extractall(output,filter='data')
    print(json.dumps(dict(collected=True,path=str(output))))
if __name__=='__main__':main()
