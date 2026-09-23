"""Authorized SSH file transport. No credentials or evaluator material enter requests."""
from pathlib import Path
import json,os,sys,time
H=Path(__file__).resolve().parent;H.relative_to(Path('/srv/encbank').resolve())
arm=sys.argv[1];assert arm in ['dense','raw_shared','comem'];R=H/('run_'+arm);B=R/'mailbox'
def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d)+'\n');t.replace(p)
for line in sys.stdin:
    d=json.loads(line);action=d.pop('action','call')
    if action=='status':
        out={n:json.loads((R/n).read_text()) for n in ['worker_ready.json','worker_failure.json','process_receipt.json'] if (R/n).exists()}
        print(json.dumps(out),flush=True);continue
    if action=='stop':
        B.mkdir(parents=True,exist_ok=True);save(B/'stop.json',d);print('{"stopped_requested":true}',flush=True);continue
    rid=d['request_id'];assert rid.replace('-','').replace('_','').isalnum()
    req=B/(rid+'.request.json');resp=B/(rid+'.response.json')
    if action=='cancel':save(B/(rid+'.cancel.json'),d);print('{"cancel_requested":true}',flush=True);continue
    assert action=='call' and not req.exists(), 'No transport blind retry or duplicate'
    assert (R/'worker_ready.json').exists() and not (R/'worker_failure.json').exists()
    B.mkdir(exist_ok=True);d['deadline_epoch']=time.time()+min(900,d.pop('remaining_seconds'))
    save(req,d)
    while not resp.exists():
        if (R/'worker_failure.json').exists():raise RuntimeError((R/'worker_failure.json').read_text())
        if time.time()>d['deadline_epoch']+60:raise TimeoutError('Worker did not acknowledge generation deadline')
        time.sleep(.15)
    print(resp.read_text().strip().replace('\n',''),flush=True)
