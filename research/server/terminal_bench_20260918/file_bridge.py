"""File-based SCP mailbox bridge; SSH only carries short control acknowledgements."""
from pathlib import Path
import hashlib,json,sys,time
H=Path(__file__).resolve().parent;H.relative_to(Path('/srv/encbank').resolve())
action,arm=sys.argv[1:3];assert arm in ['dense','raw_shared','comem'];R=H/('run_'+arm);B=R/'mailbox'
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d)+'\n');t.replace(p)
if action=='status':
 out={n:json.loads((R/n).read_text()) for n in ['worker_ready.json','worker_failure.json','process_receipt.json'] if (R/n).exists()}
 if 'worker_ready.json' in out:out['worker_ready.json'].pop('admission',None)
 print(json.dumps(out))
elif action=='stop':
 assert B.exists()
 hold=H/'hold_raw_fixgit.json'
 if arm=='raw_shared' and hold.exists() and not json.loads(hold.read_text()).get('done'):
  save(R/'stop_deferred.json',dict(reason='registered zero-answer fix-git environment continuation',at=time.time()));print('{"stop_requested":false,"deferred_for_registered_cell":true}')
 else:save(B/'stop.json',dict(reason='local Harbor parent closed'));print('{"stop_requested":true}')
else:
 rid=sys.argv[3];assert rid.replace('_','').replace('-','').isalnum()
 req=B/(rid+'.request.json');resp=B/(rid+'.response.json');incoming=H/'incoming'/(rid+'.json')
 if action=='probe':
  data=incoming.read_bytes();print(json.dumps(dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),model_called=False)))
 elif action=='publish':
  d=json.loads(incoming.read_text());assert d['request_id']==rid and not req.exists()
  assert (R/'worker_ready.json').exists() and not (R/'worker_failure.json').exists() and not (R/'process_receipt.json').exists()
  d['deadline_epoch']=time.time()+min(900,d.pop('remaining_seconds'));save(req,d);print('{"published":true}')
 elif action=='cancel':save(B/(rid+'.cancel.json'),dict(cancelled=True));print('{"cancel_requested":true}')
 elif action=='wait':
  d=json.loads(req.read_text())
  while not resp.exists():
   if (R/'worker_failure.json').exists():raise RuntimeError((R/'worker_failure.json').read_text())
   if time.time()>d['deadline_epoch']+60:raise TimeoutError('Worker response deadline exceeded')
   time.sleep(.2)
  data=resp.read_bytes();print(json.dumps(dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),path=str(resp))))
 else:raise ValueError(action)
