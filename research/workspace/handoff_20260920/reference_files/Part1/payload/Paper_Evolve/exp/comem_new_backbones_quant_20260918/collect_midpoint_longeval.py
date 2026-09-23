"""Verify and collect a complete 500-question midpoint LongEval diagnostic."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = r'''
import ast,collections,datetime,json,re,subprocess
from pathlib import Path
r=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
model_index=MODEL_INDEX
model=['Qwen3.5-9B','Qwen3.8-27B'][model_index]
plan=json.loads((r/'effective_plan.json').read_text())
source=Path('/srv/encbank/comem_followups_20260913/COMem/eval/longeval.py')
tree=ast.parse(source.read_text())
nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name=='extract_prediction') or
       (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NUM_RE' for t in n.targets))]
assert len(nodes)==2
namespace={'re':re}
exec(compile(ast.Module(body=nodes),str(source),'exec'),namespace)
score_fn=namespace['extract_prediction']
records=[];evidence=[];cells=collections.defaultdict(list);seen=set()
for shard in range(4):
 ident='large-mid-m%d-s%d'%(model_index,shard)
 task=next(t for t in plan['tasks'] if t['id']==ident)
 rec=json.loads((r/'runs'/ident/'submission.json').read_text())
 parent=json.loads((r/'runs'/ident/'parent_exit.json').read_text())
 complete=json.loads((r/task['complete']).read_text())
 p=subprocess.run(['sacct','-j',rec['job'],'-n','-P','-o','JobIDRaw,State,ExitCode,End'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 assert p.returncode==0
 assert any(v[0]==rec['job'] and v[1:3]==['COMPLETED','0:0'] for v in (s.split('|') for s in p.stdout.splitlines()))
 assert parent['actual_wait'] and parent['returncode']==0 and parent['completion_exists']
 with ((r/task['complete']).parent/'predictions.jsonl').open() as f: rows=[json.loads(line) for line in f]
 assert len(rows)==complete['records']==125
 for row in rows:
  key=(row['id'],row['arm']);assert key not in seen;seen.add(key)
  assert row['status']=='ok' and row['benchmark']=='longeval' and row['arm']=='cache_r128_s4000_h16'
  score=float(score_fn(row['text'])==row['answers'][0])
  assert abs(score-row['score'])<1e-10
  cells[row['length']].append(score)
 records.extend(rows)
 evidence.append(dict(task=ident,job=rec['job'],parent_exit=parent,complete=complete,sacct=p.stdout))
assert len(records)==500 and set(cells)=={'8k','16k','32k','64k','128k'}
assert all(len(v)==100 for v in cells.values())
summary=dict(model=model,arm='cache_r128_s4000_h16',rank=128,alpha=128,step=4000,training_horizon=8000,
 fixed_final_step=8000,scope='LongEval midpoint diagnostic only; no checkpoint selection',
 expected=500,generated=500,cpu_rescored=500,complete=True,
 score=100*sum(sum(v)/len(v) for v in cells.values())/len(cells),
 cells={k:dict(n=len(v),score=100*sum(v)/len(v)) for k,v in cells.items()},
 scorer_source=str(source),scorer_regex=namespace['_NUM_RE'].pattern,
 at=datetime.datetime.now(datetime.timezone.utc).isoformat())
print(json.dumps(dict(summary=summary,evidence=evidence,records=records)))
'''


def main():
    model_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    assert model_index in (0, 1)
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-node1',
                           'timeout -k 5s 100s /usr/bin/python3 -'],
                          input=CODE.replace('MODEL_INDEX', str(model_index)),
                          text=True, capture_output=True, timeout=115)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    out = ROOT / 'delivery' / ('qwen35_midpoint_longeval' if model_index == 0 else 'qwen38_midpoint_longeval')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result['records']), encoding='utf-8')
    for key in ('summary', 'evidence'):
        (out / (key + '.json')).write_text(json.dumps(result[key], indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (out / 'collection_exit.json').write_text(json.dumps(dict(actual_subprocess_returncode=proc.returncode, cpu_only=True)) + '\n', encoding='utf-8')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
