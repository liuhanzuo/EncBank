"""One bounded storage check per heartbeat; never launches or retries GPU work."""
import datetime,json,shlex,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STATE=ROOT/'delivery/storage_failure_20260919_recurrence/stability.json'
CODE=r'''
import datetime,json,os
from pathlib import Path
r=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
checks=[]
for parent in [r/'cache',r/'results/large-final',r/'judge_gpt6_astra']:
    d=parent/'storage-health-20260919-recurrence'
    p=d/'probe.tmp';q=d/'probe.ok';v=dict(parent=str(parent))
    try:
        d.mkdir(exist_ok=True)
        with p.open('wb') as f:f.write(b'bounded-storage-health\n');f.flush();os.fsync(f.fileno())
        assert p.read_bytes()==b'bounded-storage-health\n'
        p.replace(q);assert q.read_bytes()==b'bounded-storage-health\n'
        q.unlink();d.rmdir();v['ok']=True
    except OSError as e:v.update(ok=False,errno=e.errno,error=str(e))
    checks.append(v)
print(json.dumps(dict(at=datetime.datetime.utcnow().isoformat()+'Z',checks=checks,healthy=all(v['ok'] for v in checks))))
'''
def main():
    previous=json.loads(STATE.read_text()) if STATE.exists() else {}
    now=time.time()
    if now-previous.get('checked_unix',0)<540:
        print(json.dumps(dict(skipped=True,reason='At least nine minutes between checks',previous=previous)));return
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
        'timeout -k 5s 40s /usr/bin/python3 -c '+shlex.quote(CODE)],capture_output=True,text=True,timeout=50)
    data=json.loads(p.stdout) if p.returncode==0 else dict(healthy=False,actual_returncode=p.returncode,stderr=p.stderr[-1500:])
    gap=now-previous.get('checked_unix',0)
    consecutive=previous.get('consecutive_successes',0) if 540<=gap<=1800 else 0
    data.update(checked_unix=now,local_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        consecutive_successes=consecutive+1 if data['healthy'] else 0)
    data['eligible_for_inspected_recovery']=data['consecutive_successes']>=2
    data['note']='Eligibility is not proof of GPU-node health and does not authorize blind resubmission; inspect live jobs and latest errors.'
    STATE.parent.mkdir(exist_ok=True);STATE.write_text(json.dumps(data,indent=2)+'\n')
    history=STATE.parent/('check-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    history.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data))
if __name__=='__main__':main()
