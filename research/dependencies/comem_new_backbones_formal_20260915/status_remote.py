import collections, datetime, json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text()) if p.exists() else None
launch=read(root/'launch.json') or {}
jobs=','.join(launch[k] for k in ('training_array','data_array','evaluation_array','locomo_array') if launch.get(k))
scope=read(root/'evaluation_scope.json')
status=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),launch=launch,scope=scope,training={},data={},evaluation={})
if jobs:
 for name,cmd in [('queue',['squeue','-j',jobs,'-h','-o','%i|%j|%T|%M|%R']),
                  ('accounting',['sacct','-j',jobs,'-n','-P','--format=JobID,JobName,State,ExitCode,Elapsed'])]:
  r=subprocess.run(cmd,text=True,capture_output=True);status[name]=r.stdout.strip();status[name+'_error']=r.stderr.strip()
for model in ('Qwen3.5-9B','Qwen3.8-27B'):
 t=root/'training'/model
 status['training'][model]=dict(progress=read(t/'progress.json'),complete=read(t/'complete.json'),
     failures=[read(p) for p in sorted(t.glob('failure-*.json'))])
 d=root/'samples'/model
 status['data'][model]=dict(progress=read(d/'progress.json'),complete=read(d/'complete.json'))
 status['evaluation'][model]={}
 for p in sorted((root/'results'/model).glob('shard*')):
  v=read(p/'verified_summary.json')
  status['evaluation'][model][p.name]=dict(progress=read(p/'progress.json'),complete=read(p/'complete.json'),
      verified={k:v.get(k) for k in ('verified','samples','records','oom_records','pending_semantic_judgments','scope_id')} if v else None,
      failures=[read(f) for f in sorted(p.glob('failure-*.json'))])
status['generation_shards_verified']=sum(bool(s['verified']) for ss in status['evaluation'].values() for s in ss.values())
status['expected_generation_shards']=8
status['active_scope_generation_complete']=status['generation_shards_verified']==8 and all(
 s['verified'] and s['verified']['scope_id']==scope['scope_id'] for ss in status['evaluation'].values() for s in ss.values())
locomo_scope=read(root/'locomo_scope.json')
status['locomo_scope']=locomo_scope
status['locomo_evaluation']={}
for model in ('Qwen3.5-9B','Qwen3.8-27B'):
 status['locomo_evaluation'][model]={}
 for p in sorted((root/'results_locomo'/model).glob('shard*')):
  v=read(p/'verified_summary.json')
  status['locomo_evaluation'][model][p.name]=dict(progress=read(p/'progress.json'),complete=read(p/'complete.json'),
      verified={k:v.get(k) for k in ('verified','samples','records','oom_records','pending_semantic_judgments','scope_id')} if v else None,
      failures=[read(f) for f in sorted(p.glob('failure-*.json'))])
status['locomo_shards_verified']=sum(bool(s['verified']) for ss in status['locomo_evaluation'].values() for s in ss.values())
status['judge_deferred']=not bool(locomo_scope) and ('locomo' in scope['deferred'] if scope else False)
(root/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status))
