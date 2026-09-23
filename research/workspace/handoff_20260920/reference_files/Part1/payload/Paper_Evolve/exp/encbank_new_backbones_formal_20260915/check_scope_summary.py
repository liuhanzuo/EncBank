"""CPU synthetic check: four complete benchmarks finish without any LoCoMo records."""
import json,runpy,sys,tempfile,types
from pathlib import Path

ROOT=Path(__file__).resolve().parent
scope=json.loads((ROOT/'evaluation_scope.json').read_text())
with tempfile.TemporaryDirectory(prefix='cpu_scope_fixture_',dir=ROOT) as tmp:
    root=Path(tmp).resolve()
    assert root.is_relative_to(ROOT.resolve()) and root.name.startswith('cpu_scope_fixture_')
    (root/'evaluation_scope.json').write_text(json.dumps(scope))
    stub=types.ModuleType('common');stub.ROOT=root;stub.MODELS=[{'name':'CPU_FIXTURE','j':6}]
    stub.ARMS=['CPU_FIXTURE_ARM'];stub.SHARDS=4
    def dump(p,v):Path(p).write_text(json.dumps(v))
    stub.dump=dump
    fake_scope=types.ModuleType('evaluation_scope');fake_scope.read_scope=lambda:scope
    before={k:sys.modules.get(k) for k in ('common','evaluation_scope')}
    sys.modules['common']=stub;sys.modules['evaluation_scope']=fake_scope
    paths=[];seq=0
    for shard in range(4):
        d=root/'results/CPU_FIXTURE'/f'shard{shard}';d.mkdir(parents=True)
        (d/'verified_summary.json').write_text(json.dumps({'verified':True,'scope_id':scope['scope_id']}))
        paths.append(d/'predictions.jsonl')
    handles=[p.open('w') for p in paths]
    try:
        for b,counts in [('ruler',[100]*15),('longeval',[100]*5),('longbench',[200]*5+[150]),('babilong',[100]*21)]:
            for cell,n in enumerate(counts):
                for i in range(n):
                    row=dict(id=f'{b}:{cell}:{i}',benchmark=b,task=f'task{cell}',length='fixture',
                             arm='CPU_FIXTURE_ARM',status='ok',score=.5)
                    handles[seq%4].write(json.dumps(row)+'\n');seq+=1
    finally:
        for f in handles:f.close()
    try:
        # Suppress the ordinary summary print for synthetic fixture records.
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(ROOT/'summarize_remote.py'))
        good=json.loads((root/'summary.json').read_text())
        assert seq==5250 and good['all_metrics_complete'] and good['generation_verified_complete']
        table=good['models']['CPU_FIXTURE']['table']['CPU_FIXTURE_ARM']
        assert set(table)==set(scope['benchmarks']) and all(v['score']==50 for v in table.values())
        # Missing/OOM scientific outputs must not silently reduce the denominator.
        rows=[json.loads(x) for x in paths[0].read_text().splitlines()]
        rows[0].update(status='OOM',score=None)
        paths[0].write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(ROOT/'summarize_remote.py'))
        bad=json.loads((root/'summary.json').read_text())
        assert not bad['all_metrics_complete']
        assert bad['models']['CPU_FIXTURE']['table']['CPU_FIXTURE_ARM']['ruler']['score'] is None
    finally:
        for k,v in before.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v
    # TemporaryDirectory cleanup targets only this checked, task-local fixture.
    assert root.is_relative_to(ROOT.resolve()) and root.name.startswith('cpu_scope_fixture_')
report={'passed':True,'synthetic_test_only':True,'full_four_benchmark_completion_without_locomo':True,
        'oom_remains_unscored':True,'synthetic_inputs':5250}
(ROOT/'scope_summary_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
