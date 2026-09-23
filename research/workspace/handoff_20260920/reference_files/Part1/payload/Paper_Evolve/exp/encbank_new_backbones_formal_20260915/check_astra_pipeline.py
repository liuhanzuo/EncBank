"""Mock transport tests for resume, exact-input reuse, OOM, and invalid labels."""
import ast,json,tempfile,threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from astra_judge_batch import ROOT,MODEL,run_batch
import locomo_judge_worker as worker
for name in ('astra_judge_batch.py','locomo_judge_worker.py','export_locomo_judge.py'):
    ast.parse((ROOT/name).read_text())
protocol=dict(model=MODEL,protocol_id='SYNTHETIC_TEST_ONLY',status='READY',calibration_passed=True)
calls=[];mutex=threading.Lock()
def fake(stim,out):
    with mutex:calls.append(stim)
    return dict(ok=True,answer='CORRECT',usage=None)
with tempfile.TemporaryDirectory(prefix='synthetic_astra_check_',dir=ROOT) as td:
    temp=Path(td).resolve();assert temp.is_relative_to(ROOT.resolve())
    rows=[dict(cohort='synthetic',arm='a',id='1',shard=0,category=1,status='ok',question='Q',answers=['A'],pred='A'),
          dict(cohort='synthetic',arm='b',id='1',shard=0,category=1,status='ok',question='Q',answers=['A'],pred='A'),
          dict(cohort='synthetic',arm='a',id='2',shard=0,category=5,status='ok',question='Q2',answers=['X'],pred=''),
          dict(cohort='synthetic',arm='a',id='3',shard=0,category=1,status='OOM',question='Q3',answers=['A'],pred=None)]
    source=temp/'inputs.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    r=run_batch([source],temp/'out',protocol,workers=3,classifier=fake)
    assert r['available_complete'] and r['decisions']==4 and r['oom']==1 and len(calls)==1
    assert all(c['score'] is None for arms in r['summary'].values() for c in arms.values())
    r=run_batch([source],temp/'out',protocol,workers=3,classifier=fake)
    assert r['decisions']==4 and len(calls)==1
    bad=temp/'bad.jsonl';bad.write_text(json.dumps(dict(rows[0],id='4',pred='other'))+'\n')
    r=run_batch([bad],temp/'failed',protocol,workers=1,classifier=lambda s,o:dict(ok=True,answer='MAYBE'))
    assert not r['available_complete'] and r['decisions']==0 and len(r['errors'])==1
    r=run_batch([bad],temp/'failed',protocol,workers=1,classifier=fake)
    assert r['available_complete'] and r['decisions']==1
    # Exercise the actual collector-to-judge handoff, including a resumed file
    # that was downloaded before the worker stopped. No network or API calls.
    worker_root=temp/'collector';worker_root.mkdir()
    (worker_root/'judge_protocol.json').write_text(json.dumps(protocol))
    worker_out=worker_root/'out';(worker_out/'inputs').mkdir(parents=True)
    cached=worker_out/'inputs'/'synthetic_shard0.jsonl'
    cached.write_text(json.dumps(rows[0])+'\n')
    manifest=dict(verified_shards=1,expected_shards=8,files=[dict(file=cached.name,records=1)])
    handoffs=[]
    def fake_batch(inputs,out,selected_protocol,workers):
        assert inputs==[cached] and inputs[0].is_file()
        assert selected_protocol==protocol and workers==8
        handoffs.append(inputs)
        (out/'STOP').touch()
        return dict(errors=[],decisions=1,available_complete=True,oom=0)
    with patch.object(worker,'ROOT',worker_root),patch.object(worker,'OUT',worker_out), \
         patch.object(worker.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(manifest))) as transport, \
         patch.object(worker,'run_batch',side_effect=fake_batch),patch.object(worker.time,'sleep'):
        worker.main()
    assert len(handoffs)==1 and transport.call_count==1
    assert json.loads((worker_out/'worker_status.json').read_text())['phase']=='STOPPED'
report=dict(passed=True,scope='Mock software checks; no experimental measurements',
    verified=['independent prompts','identical-input reuse','resume without duplicate calls','OOM stays null','partial scores stay null','invalid labels unscored and retryable','collector hands downloaded Path files to judge on resume'])
(ROOT/'astra_pipeline_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
