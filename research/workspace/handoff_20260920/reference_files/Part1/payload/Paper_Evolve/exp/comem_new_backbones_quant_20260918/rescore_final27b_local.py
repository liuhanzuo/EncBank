"""Rescore saved complete answers using the original scoring function ASTs, CPU only."""
import ast, collections, datetime, hashlib, json, math, re, string
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'delivery/qwen38_final_generation'
SOURCE = OUT / 'scoring_sources'

def load_functions(filename, names, assignments=()):
    path = SOURCE / filename
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name in names) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments for t in n.targets))]
    assert len(nodes) == len(names) + len(assignments)
    scope = dict(re=re, collections=collections, string=string)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope

ruler = load_functions('ruler.py', ['_string_match_all_one'])
longeval = load_functions('longeval.py', ['extract_prediction'], ['_NUM_RE'])
longbench = load_functions('longbench.py', ['normalize_answer', 'compute_f1', 'compute_f1_multi'])
babilong = load_functions('babilong_metrics.py', ['preprocess_output', 'compare_answers'], ['TASK_LABELS'])
rows = [json.loads(line) for line in (OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
votes = [json.loads(line) for line in (OUT/'judge_decisions.jsonl').read_text(encoding='utf-8').splitlines()]
judge = {v['id']: v for v in votes}
assert len(rows) == len({r['id'] for r in rows}) == 7236
assert len(votes) == len(judge) == 1986
assert all(v['cohort'] == 'Qwen3.8-27B' and v['arm'] == 'cache_r128_s8000_h16' and v['status']=='ok'
           and v['model'] == 'gpt-6-astra' and v['protocol_id'] == 'midcache-locomo-gpt6-astra-v1'
           and v['judge_correct'] in (0,1) for v in votes)
expected = dict(ruler=1500, longeval=500, longbench=1150, babilong=2100, locomo=1986)
cells = collections.defaultdict(lambda: collections.defaultdict(list))
rescored = 0
for row in rows:
    assert row['status']=='ok' and row['rank']==128 and row['step']==8000 and row['arm']=='cache_r128_s8000_h16'
    b, text = row['benchmark'], row['text']
    if b == 'ruler':
        value = ruler['_string_match_all_one'](text, row['answers'])
    elif b == 'longeval':
        value = float(longeval['extract_prediction'](text) == row['answers'][0])
    elif b == 'longbench':
        value = longbench['compute_f1_multi'](text, row['answers'])
    elif b == 'babilong':
        value = float(babilong['compare_answers'](row['extra']['target'], text, row['extra']['question'], babilong['TASK_LABELS'][row['task']]))
    else:
        assert b == 'locomo'
        vote = judge[row['id']]
        assert int(vote['category']) == int(row['extra']['category'])
        value = vote['judge_correct']
    if b != 'locomo':
        assert math.isclose(value, row['score'], abs_tol=1e-10), (row['id'], value, row['score'])
        rescored += 1
    cells[b][(row['task'], row['length'])].append(value)
assert rescored == 5250
assert set(judge) == {r['id'] for r in rows if r['benchmark']=='locomo'}
scores, detail = {}, {}
for benchmark, n in expected.items():
    data = cells[benchmark]
    assert sum(map(len, data.values())) == n
    if benchmark == 'locomo':
        value = 100 * sum(map(sum, data.values())) / n
    else:
        assert len(data) == dict(ruler=15, longeval=5, longbench=6, babilong=21)[benchmark]
        value = 100 * sum(sum(v)/len(v) for v in data.values()) / len(data)
    scores[benchmark] = value
    detail[benchmark] = dict(expected=n, generated=n, scored=n, complete=True, score=value,
                             cells={str(k):dict(n=len(v), score=100*sum(v)/len(v)) for k,v in data.items()})
scores['avg'] = sum(scores[b] for b in expected)/5
category = {}
for name, categories, n in [('C1-4',{1,2,3,4},1540), ('C5',{5},446)]:
    part = [v for v in votes if int(v['category']) in categories]
    assert len(part) == n
    category[name] = dict(n=n, score=100*sum(v['judge_correct'] for v in part)/n)
receipt = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
assert receipt['generation_complete'] and all(v['actual_wait'] and v['returncode']==0 for v in [x['parent_exit'] for x in receipt['evidence']])
report = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), model='Qwen3.8-27B',j=21,
              rank=128,alpha=128,step=8000,scores=scores,benchmarks=detail,locomo_categories=category,
              cpu_rescored=rescored,judge_records=1986,errors=0,full_five_benchmark_complete=True,
              scoring='Original function ASTs executed unmodified on CPU; every automatic score matches saved generation score.',
              source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')},
              generation_evidence='summary.json; canonical_sync_result.json; actual parent_wait and Slurm0:0 for all four shards')
(OUT/'five_benchmark_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(scores=scores,cpu_rescored=rescored,judge_records=1986,locomo_categories=category)))
