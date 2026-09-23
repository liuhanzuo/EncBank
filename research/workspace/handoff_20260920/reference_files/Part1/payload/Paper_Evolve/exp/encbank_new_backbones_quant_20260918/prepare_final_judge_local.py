"""Prepare remaining authorized judgments on controller-local disk; never calls the API."""
import json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CODE=r'''
import datetime,hashlib,json,os,shutil,sys
from pathlib import Path
r=Path('/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918')
old=Path('/srv/encbank/encbank_new_backbones_formal_20260915')
stage=Path('/tmp/qcm-q18-final-judge-20021-20260919-0840')
assert not stage.exists(),'Prepare already attempted; inspect before repeating'
assert shutil.disk_usage('/tmp').free>3*2**30
status=json.loads((r/'status.json').read_text());assert status['phase']=='GENERATION_COMPLETE' and status['completed']==28 and status['failed']==0
for pid,needle in [(1599472,b'control/coordinator.py'),(2005194,b'judge_watch.py')]:
 p=Path('/proc')/str(pid)/'cmdline';assert not(p.exists() and needle in p.read_bytes())
judge=r/'judge_gpt6_astra'
saved=(judge/'judge_decisions.jsonl').read_bytes();votes=[json.loads(l) for l in saved.splitlines()]
seen={(v['cohort'],v['arm'],v['id']):v for v in votes};assert len(seen)==len(votes)==10427
stage.mkdir(mode=0o700);out=stage/'judge_gpt6_astra';out.mkdir();(out/'inputs').mkdir();(stage/'tmp').mkdir()
hashes={}
for name,source in [('astra_judge_batch.py',old),('codex_judge_linux.py',old),('locomo_judge_reference.py',old),('judge_watch.py',r)]:
 data=(source/name).read_bytes();(stage/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
sys.path.insert(0,str(stage));from astra_judge_batch import identity
protocol=json.loads((judge/'protocol.json').read_text());assert protocol['protocol_id']=='midcache-locomo-gpt6-astra-v1' and protocol['model']=='gpt-6-astra' and protocol['status']=='READY' and protocol['calibration_passed']
(out/'protocol.json').write_bytes((judge/'protocol.json').read_bytes());(out/'judge_decisions.jsonl').write_bytes(saved)
if (judge/'transport_recovery.json').exists():shutil.copy2(str(judge/'transport_recovery.json'),str(out/'transport_recovery.json'))
plan=json.loads((r/'plan.json').read_text());inputs=[];allrows=[]
for task in plan['tasks']:
 if not task['id'].startswith(('quant-full-','large-final-')):continue
 assert status['tasks'][task['id']]['state']=='COMPLETED'
 shard=int(task['id'].rsplit('s',1)[1]);model_index=0 if task['model']=='Qwen3.5-9B' else 1
 reference=judge/'inputs'/('quant-full-m%d-s%d.jsonl'%(model_index,shard))
 questions={v['id']:v for v in map(json.loads,reference.read_text().splitlines())}
 rows=[]
 for line in ((r/task['complete']).parent/'predictions.jsonl').open():
  v=json.loads(line)
  if v['benchmark']!='locomo':continue
  q=questions[v['id']]
  assert v['status']=='ok' and int(v['extra']['category'])==int(q['category']) and v['answers']==q['answers']
  rows.append(dict(cohort=task['model'],arm=v['arm'],id=v['id'],shard=shard,category=q['category'],status=v['status'],question=q['question'],answers=q['answers'],pred=v['text']))
 prior=judge/'inputs'/(task['id']+'.jsonl')
 if prior.exists():assert [json.loads(l) for l in prior.read_text().splitlines()]==rows
 target=out/'inputs'/(task['id']+'.jsonl');target.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows))
 inputs.append(str(target));allrows.extend(rows)
assert len(inputs)==16 and len(allrows)==11916
assert len({(v['cohort'],v['arm'],v['id']) for v in allrows})==11916
copied=0;remaining=set()
for v in allrows:
 key=(v['cohort'],v['arm'],v['id']);digest=identity(v,protocol)
 if key in seen:
  assert seen[key]['stimulus_digest']==digest;continue
 remaining.add(digest);dest=out/'cache'/digest
 if dest.exists():continue
 src=judge/'cache'/digest
 if src.exists():shutil.copytree(str(src),str(dest));copied+=1
 else:
  src=old/'judge_gpt6_astra/formal_new_models/cache'/digest
  if (src/'decision.json').exists():
   dest.mkdir(parents=True);shutil.copy2(str(src/'decision.json'),str(dest/'decision.json'));copied+=1
 if dest.exists():
  for dispatch in dest.rglob('dispatch.json'):assert (dispatch.parent/'result.json').exists(), 'Uncertain prior request; inspect, do not resend'
  for attempt in dest.glob('attempt-*'):
   if attempt.is_dir():assert list(attempt.rglob('result.json')),'Uncertain prior attempt; inspect, do not resend'
  decision=dest/'decision.json'
  if decision.exists():
   vote=json.loads(decision.read_text());assert vote['stimulus_digest']==digest and vote['protocol_id']==protocol['protocol_id'] and vote['model']=='gpt-6-astra'
(stage/'inputs.json').write_text(json.dumps(inputs,indent=2))
info=dict(at=datetime.datetime.utcnow().isoformat()+'Z',stage=str(stage),out=str(out),inputs=16,records=11916,
 preserved_decisions=10427,remaining_records=1489,remaining_unique_stimuli=len(remaining),copied_cache_entries=copied,
 source_hashes=hashes,source_decisions_sha256=hashlib.sha256(saved).hexdigest(),api_calls=0,workers=4,
 scientific_protocol_unchanged=True,canonical_generation_complete=True)
(stage/'prepared.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
                  '/srv/encbank/Paper_Evolve/.venv/bin/python -I -B -c '+shlex.quote(CODE)],
                 capture_output=True,text=True,encoding='utf-8',timeout=58)
out=ROOT/'delivery/final_judge_local';out.mkdir(exist_ok=True)
(out/'prepare_stdout.txt').write_text(p.stdout,encoding='utf-8');(out/'prepare_stderr.txt').write_text(p.stderr,encoding='utf-8')
assert p.returncode==0,p.stderr
v=json.loads(p.stdout);(out/'prepared.json').write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v))
