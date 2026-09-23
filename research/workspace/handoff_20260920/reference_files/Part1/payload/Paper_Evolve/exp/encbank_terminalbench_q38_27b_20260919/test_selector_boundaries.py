"""CPU-only first-turn and sparse-history selector regression checks."""
import json,math
from selection import lexical,nucleus
from memory_selectors import bm25_scores

checks=[]
for arm in ['iter_k12','iter_k48','bm25_p90','iter48_qk_p90']:
    ids,detail=lexical([],[7,9],arm)
    assert ids==[]
    checks.append(dict(arm=arm,case='empty history',selected=ids))
ids,detail=lexical([[1,2],[3,4]],[9],'bm25_p90')
assert ids==[0,1] and detail['zero_mass'] and not detail['target_met']
checks.append(dict(arm='bm25_p90',case='zero overlap, fewer than minimum',selected=ids))
docs=[[1,2],[1,1,3],[4,5],[1,6],[7,8]]
ids,detail=lexical(docs,[1],'bm25_p90')
assert detail==nucleus(bm25_scores(docs,[1]),list(range(len(docs))))
assert ids==detail['selected'] and len(ids)>=4 and detail['target_met']
detail=nucleus([1.]*60,list(range(60)))
assert len(detail['selected'])==48 and detail['cap_limited'] and not detail['target_met']
for invalid in [[float('nan')],[-1.],None]:
    try:nucleus(invalid,[0])
    except (AssertionError,TypeError):pass
    else:raise AssertionError('Invalid nonempty scores accepted')
print(json.dumps(dict(passed=True,checks=checks,positive_pool_unchanged=True,cap_unmet_recorded=True,invalid_scores_rejected=True,model_calls=0)))
