"""CPU-only routing/support regression checks; these are not benchmark results."""
import ast,json,os
from pathlib import Path
from evaluation_scope import read_scope,results_directory,shard_count
root=Path(__file__).resolve().parent
for name in ('evaluate.py','verify.py','evaluation_scope.py','status_remote.py','submit_locomo.py'):
    ast.parse((root/name).read_text())
assert read_scope()['scope_id']=='four-benchmarks-locomo-deferred-20260915'
assert results_directory()=='results'
spec={'shards':4,'samples':7236,'cells':{'ruler:test':5250,'locomo:qa':1986}}
four=[shard_count(spec,s,read_scope()) for s in range(4)]
assert four==[1313,1313,1312,1312]
os.environ['MIDCACHE_SCOPE_FILE']='locomo_scope.json'
os.environ['MIDCACHE_RESULTS_DIR']='results_locomo'
assert results_directory()=='results_locomo'
locomo=[shard_count(spec,s,read_scope()) for s in range(4)]
assert locomo==[496,496,497,497]
assert all(a+b==1809 for a,b in zip(four,locomo))
os.environ['MIDCACHE_RESULTS_DIR']='results'
try:results_directory()
except AssertionError:pass
else:raise AssertionError('Cross-scope output collision was allowed')
report=dict(passed=True,four=four,locomo=locomo,locomo_records=27804,scopes_disjoint=True,
    note='Synthetic routing checks only; no scientific scores')
(root/'locomo_scope_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
