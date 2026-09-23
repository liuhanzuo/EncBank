"""Inspect this campaign's new storage failure without retrying scientific tasks."""
import json, shlex, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CODE=r'''
import datetime,json,os,subprocess,uuid
from pathlib import Path
r=Path('/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918')
tasks=['large-final-m0-s3']+['large-final-m1-s%d'%i for i in range(4)]
result=dict(at=datetime.datetime.utcnow().isoformat()+'Z',tasks={})
for task in tasks:
    d=r/'runs'/task
    item={}
    for name in ['submission.json','start.json','parent_exit.json']:
        p=d/name
        item[name]=json.loads(p.read_text()) if p.exists() else None
    for name in ['child.stderr.log','parent.stderr.log']:
        p=d/name
        if p.exists():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));item[name]=f.read().decode('utf-8','replace')
    result['tasks'][task]=item
result['processes']={}
for label,pid,needle in [('owner',548826,b'control/coordinator.py'),('judge',2904295,b'judge_watch.py'),('summary',337576,b'summarize.py')]:
    p=Path('/proc')/str(pid)/'cmdline'
    result['processes'][label]=dict(pid=pid,alive=p.exists() and needle in p.read_bytes())
health=[]
for parent in [r/'control',r/'results/large-final',r/'judge_gpt6_astra']:
    p=parent/('health-20260919-'+uuid.uuid4().hex)
    item=dict(parent=str(parent));renamed=p.with_suffix('.ok')
    try:
        with p.open('xb') as f:f.write(b'campaign-storage-probe\n');f.flush();os.fsync(f.fileno())
        assert p.read_bytes()==b'campaign-storage-probe\n'
        p.rename(renamed);assert renamed.read_bytes()==b'campaign-storage-probe\n'
        renamed.unlink();item['ok']=True
    except OSError as e:item.update(ok=False,errno=e.errno,error=str(e))
    health.append(item)
result['health']=health
result['healthy']=all(x['ok'] for x in health)
print(json.dumps(result))
'''
def main():
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
        'timeout -k 5s 40s /usr/bin/python3 -c '+shlex.quote(CODE)],capture_output=True,text=True,timeout=50)
    assert p.returncode==0,p.stderr
    data=json.loads(p.stdout)
    out=ROOT/'delivery/storage_failure_20260919';out.mkdir(exist_ok=True)
    (out/'inspection.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(at=data['at'],health=data['health'],processes=data['processes'])))
if __name__=='__main__':main()
