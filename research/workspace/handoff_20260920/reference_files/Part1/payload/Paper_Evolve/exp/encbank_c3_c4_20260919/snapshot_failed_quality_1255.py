"""Preserve real failed outputs before node-local continuation; no shared writes."""
import hashlib,json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery/storage_failure_20260919_1255'
CODE=r'''
import datetime,json,subprocess
from pathlib import Path
p=Path('/srv/encbank/encbank_c3_c4_20260919')
jobs={6:'107555',12:'107601',18:'107592'}
acct=subprocess.check_output(['sacct','-j',','.join(jobs.values()),'-n','-P','-o','JobIDRaw,State,ExitCode,NodeList,End'],universal_newlines=True)
for j,job in jobs.items():assert job+'|FAILED|1:0|' in acct
f=Path('/proc/3454603/cmdline');assert not f.exists() or b'eval_owner.py' not in f.read_bytes()
records=[]
for j,job in jobs.items():
    out=p/'quality'/('j%d_s42'%j)
    assert not (out/'complete.json').exists()
    files={}
    for name in ['predictions.jsonl','progress.json','protocol.json','correctness.json','cache_location.json','parent_exit.json']:
        path=out/name
        if path.exists():files[name]=path.read_text()
    files['stderr.log']=(p/'logs'/('eval-j%d-%s.err'%(j,job))).read_text()
    assert '[Errno 121]' in files['stderr.log']
    records.append(dict(j=j,previous_job=job,files=files))
print(json.dumps(dict(at=datetime.datetime.utcnow().isoformat()+'Z',sacct=acct,records=records,
    owner_stderr=(p/'eval_owner.stderr.log').read_text(),canonical_owner_dead=True)))
'''
def main():
    assert not OUT.exists(),'Already snapshotted: inspect instead of repeating'
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(CODE)],capture_output=True,encoding='utf-8',timeout=40)
    assert r.returncode==0,r.stderr
    data=json.loads(r.stdout);OUT.mkdir(parents=True)
    tasks=[]
    for rec in data['records']:
        dest=OUT/('j%d'%rec['j']);dest.mkdir()
        for name,text in rec.pop('files').items():(dest/name).write_text(text,encoding='utf-8',newline='')
        raw=(dest/'predictions.jsonl').read_bytes();assert raw.endswith(b'\n')
        rows=[json.loads(line) for line in raw.splitlines()]
        assert len(rows)==len({(x['id'],x['arm']) for x in rows})
        assert all(x['j']==rec['j'] and x['status']=='ok' for x in rows)
        tasks.append(dict(rec,saved_records=len(rows),sha256=hashlib.sha256(raw).hexdigest(),node='gpu-node4' if rec['j']==18 else 'gpu-node1'))
    data['tasks']=tasks;(OUT/'snapshot.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tasks=tasks,sacct=data['sacct'])))
if __name__=='__main__':main()
