"""One actual SSH snapshot; metadata only, no model or scoring."""
import json
from pathlib import Path
from datetime import datetime, timezone
R=Path('/data/liuhanzuo/encbank_v2_20260908');B=R/'workspace/exp/encbank_v2_benchmarks_20260908';O=R/'outputs/benchmarks_v2/full'
p=B/'heartbeat_extended_20260909_0931_snapshot.json'
assert not p.exists()
state=json.loads((R/'outputs/queue_v2/full/state.json').read_text())
d={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'state':state,
   'eligible_native_markers':sorted(str(p) for family in ('longeval','infinitebench','locomo') for arm in ('pub','pub_sink','cbos') for p in (O/family/arm).glob('*/COMPLETED.json')),
   'eligible_ruler_files':sorted(str(p) for arm in ('pub','pub_sink','cbos') for p in (O/'ruler'/arm).glob('*.json'))}
p.write_text(json.dumps(d,indent=2)+'\n')
known=set()
for name in ('0129','0229','0329','0429','0531','0631','0731','0831'):
    for c in json.loads((B/f'heartbeat_extended_20260909_{name}.json').read_text())['writeback']:
        known.add((c['benchmark'],c['task'],c.get('length'),c['arm']))
new=[]
for f in d['eligible_ruler_files']:
    o=json.loads(Path(f).read_text());a=o['args'];arm=o['arms'][0];task=a['tasks'];length=a['lengths']
    job=state['jobs'].get(f'ruler__{arm}__{task}_{length}',{})
    if ('ruler',task,length,arm) not in known and job.get('status')=='completed' and job.get('exit_code')==0:new.append([arm,task,length,len(o['rows'])])
locomo={}
for arm in ('pub','pub_sink','cbos'):
    locomo[arm]=[int(Path(x).parent.name.split('_s')[1].split('of')[0]) for x in d['eligible_native_markers'] if f'/locomo/{arm}/' in x]
print(json.dumps({'timestamp_utc':d['timestamp_utc'],'main_completed':sum(j.get('status')=='completed' for j in state['jobs'].values()),'main_total':len(state['jobs']),'new_ruler_candidates':new,'locomo_shards':locomo,'snapshot':str(p)}),flush=True)
