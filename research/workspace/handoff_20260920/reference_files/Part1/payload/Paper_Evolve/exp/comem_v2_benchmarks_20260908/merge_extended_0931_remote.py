"""Assemble independently rescored new cells from the fixed 09:31 snapshot."""
import json
from pathlib import Path
from datetime import datetime, timezone
R=Path('/data/liuhanzuo/comem_v2_20260908'); B=R/'workspace/exp/comem_v2_benchmarks_20260908'
def read(p):return json.loads(p.read_text())
s=read(B/'heartbeat_extended_20260909_0931_snapshot.json')
r=read(B/'heartbeat_extended_20260909_0931_ruler.json')
assert s['timestamp_utc']==r['snapshot_timestamp_utc']
assert r['checks']['new_ruler_raw_rescores']==100
assert r['checks']['old_outputs_rescored']==r['checks']['longeval_rescores']==r['checks']['models_constructed']==0
assert r['checks']['torch_threads']==2 and r['checks']['torch_interop_threads']==16 and not r['checks']['cuda_initialized']
cells=r['writeback']
assert {(c['task'],c['length'],c['arm']) for c in cells}=={('variable_tracking','128k','cbos'),('niah_single_2','256k','pub')}
paired=[]
for c in cells:
    assert c['complete'] and c['n']==c['expected_n']==50
    raw=read(Path(c['source']))['rows']
    counts={len(x['answers']) for x in raw}
    assert len(counts)==1
    targets_per_example=counts.pop()
    got=[]
    for x in raw:
        value=x[c['arm']+'_recall']*targets_per_example
        assert abs(value-round(value))<1e-8
        got.append(round(value))
    numerator=sum(got); denominator=50*targets_per_example
    assert abs(100*numerator/denominator-c['score_percent'])<1e-8
    c.update(raw_mean=c['score_percent']/100, exact_fraction={'matched_answer_targets':numerator,'total_answer_targets':denominator,'targets_per_example':targets_per_example,'mean_recall':numerator/denominator},
             accuracy_if_single_target_only=(numerator/denominator if targets_per_example==1 else None))
    for arm,p in c['paired_contrasts'].items():
        assert p['new_arm_wins']+p['ties']+p['new_arm_losses']==50
        paired.append({'benchmark':'ruler','task':c['task'],'length':c['length'],'new_arm':c['arm'],'reference_arm':arm,**p})
pending=[c for c in r['pending'] if c['benchmark']=='locomo']
assert len(pending)==3 and all(c['n']==1788 and c['expected_n']==1986 and not c['coverage_complete'] for c in pending)
assert all(not cat['coverage_complete'] and cat['score_percent'] is None for c in pending for cat in c['categories'])
assert all(c['completed_shards']==list(range(9)) for c in pending)
ruler_waiting=[]
for name,j in s['state']['jobs'].items():
    if name.startswith(('ruler__pub__','ruler__pub_sink__','ruler__cbos__')) and name.endswith(('_128k','_256k')) and j.get('status')!='completed':
        ruler_waiting.append({'job':name,'status_at_snapshot':j.get('status'),'complete':False,'score_percent':None})
assert len(ruler_waiting)==5
limits=[
 'RULER legacy metadata/CLI/seed pairing only: original full input tokens, selected runtime pack, and generation-cache keys were not persisted and cannot be independently reconstructed from this artifact.',
 'RULER scores use the original CoMem adaptation answer-substring recall; variable_tracking is five-needle retrieval, not fact-composition accuracy. Equal target counts permit the exact matched-target fraction, not whole-example accuracy.',
 'Reference scores are previously independently audited and are not rescored this round; paired counts are descriptive and no significance test is claimed.',
 'LoCoMo only counts snapshot-eligible complete shards and complete categories/tasks; all new category and task scores remain null. Category denominators are 282,321,96,841,446, totaling 1986 distinct outputs per arm.',
 'LongEval and all five MC methods were complete and audited earlier, and were not rescored. Missing historical LongEval runtime-pack evidence remains missing.',
 'Remote accuracy audits do not establish local5090 timing, peak memory, or architecture causal effects.'
]
d={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'snapshot_timestamp_utc':s['timestamp_utc'],
   'snapshot_file':str(B/'heartbeat_extended_20260909_0931_snapshot.json'),
   'scope':'Only newly complete extended baseline cells since 08:31; fixed actual 09:33:50 snapshot, no later completions included.',
   'execution':{'location':'remote CPU','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':2,'MKL_NUM_THREADS':2,'torch_threads':2,'torch_interop_threads':16,'models_constructed':0,'cuda_initialized':False},
   'writeback':cells,'new_complete_cells':2,'new_predictions_audited':100,'paired_counts':paired,
   'pending':pending,'coverage':pending,'pending_ruler_128k_256k':ruler_waiting,
   'checks':r['checks'],'protocol_limits':limits,
   'negative_results':[
    'Isolated KV five-needle128k mean recall60.8 is below previously audited V2/replay93.2/100, but isolated KV wins3 of50 metadata-paired examples against V2 (ties3, losses44). The per-example exception is retained.',
    'Original CoMem single256k34 is below the previous V2/replay100/100; no general length-monotonicity or causal attribution is established.',
    'LoCoMo three baseline arms each remain1788/1986, with no complete category; partial means are excluded.'
   ],'evidence_files':['heartbeat_extended_20260909_0931_ruler.json','heartbeat_extended_20260909_0931_snapshot.json']}
assert sum(c['n'] for c in cells)==100
for suffix,value in [('.json',d),('_writeback.json',{'schema_version':1,'snapshot_timestamp_utc':s['timestamp_utc'],'new_complete_cells':cells,'new_complete_cell_count':2,'new_predictions_audited':100,'paired_counts':paired,'pending':pending,'pending_ruler_128k_256k':ruler_waiting,'protocol_limits':limits,'evidence':str(B/'heartbeat_extended_20260909_0931.json')})]:
    p=B/('heartbeat_extended_20260909_0931'+suffix);assert not p.exists();p.write_text(json.dumps(value,indent=2)+'\n')
md=['# 09:31 扩展任务基线独立核验','',f"固定实际远端快照：{s['timestamp_utc']} UTC（北京时间09:33:50）；main296/374。只审08:31之后、该快照前完整结束的新增单元，不追收后续输出。",'',
    '本轮仅2个新RULER格、100个新预测。远端CPU执行原始评分复算及metadata/CLI/seed配对，Torch2线程、interop16、OMP2/MKL2、CUDA隐藏且未初始化，零模型加载。没有修改论文、根计划、队列或同步器；已核MC和LongEval均未重评分。','',
    '| Task | Length | New arm | n | Score (%) | Exact matched-target fraction |', '|---|---|---|---:|---:|---|']
for c in cells:
    f=c['exact_fraction'];md.append(f"| {c['task']} | {c['length']} | {c['arm']} | 50 | {c['score_percent']:.2f} | {f['matched_answer_targets']}/{f['total_answer_targets']} |")
md+=['','pub=原始CoMem，cbos=isolated full KV，不能以cbos代表CacheBlend。256k单针为17/50；128k五针为152/250个答案目标的平均substring recall，不是152个独立问题，也不是whole-example accuracy。','',
     '100条新raw输出通过原CoMem RULER评分复算、连续i0–49、完整summary、无OOM、模型/reader参数、远端准入和PYTHONHASHSEED日志核对。与V2/j0旧已核记录的task/length/i/answers/n_tokens及CLI参数逐例一致。旧参考评分没有重新调用。','',
     '同stock Qwen3-8B、j12、chunk512、top12、seed42。单针使用BM25和cap48；五针使用iter_BM25、hop4以及VT有效cap60（原CLI为48）。历史raw没有保存完整tokens、运行时有序selected pack或generation-cache精确键，因此这里只能声明metadata/CLI/seed配对，不能升级为exact-input/pack配对或跨runtime严格等价。','',
     '## 逐例描述性差异','',
     '方向为新基线相对旧参考，n50；没有显著性推断。','',
     '| Task / length | Baseline | Reference | Wins | Ties | Losses | Mean delta (pp) |','|---|---|---|---:|---:|---:|---:|']
for p in paired:md.append(f"| {p['task']} / {p['length']} | {p['new_arm']} | {p['reference_arm']} | {p['new_arm_wins']} | {p['ties']} | {p['new_arm_losses']} | {p['new_arm_minus_reference_pp']:.2f} |")
md+=['','isolated KV五针128k平均60.8%，低于此前V2的93.2%和Replay的100%；但它有3例得分超过V2，保留该逐例不利结果。CoMem单针256k34%，此前V2和Replay均100%；不同长度不能据此推断单调下降或隔离因果。','',
     '## LoCoMo：覆盖未齐，不报局部均值','',
     '三臂pub/pub_sink/cbos均仅完整shard0–8，共1788/1986。每臂五类覆盖一致：','',
     '| Category | Completed n | Formal n | Eligible score |','|---|---:|---:|---|']
for c in pending[0]['categories']:md.append(f"| {c['category']} | {c['n']} | {c['expected_n']} | — |")
md+=['','覆盖核对完整receipt、run_config期望index/id/task、分片连续索引、跨分片唯一性及category计数；未重新评分这些部分输出。五类均未全，所有新增LoCoMo分数保持null。未来每臂完整任务应核1986个不同输出，并拆成五类，不能把五类分母当成5×1986。','',
     '剩余RULER未完整项仅256k：BOS/isolated单针，以及三臂multi-key，共5格。MC五方法与LongEval全矩阵此前已完整并通过核验，本轮零重评分；旧LongEval缺runtime pack限制不变。','',
     '## 紧凑写回建议','',
     '只填既有附录RULER基线矩阵：Single-needle/256k原版34.0；Five-needle/128k isolated KV60.8。后一行三基线因此齐为0/4.4/60.8，V2/Replay仍为旧已核93.2/100。LoCoMo不补任何分数，不新增正文表；其他未完整格保持空白。','',
     '机器文件：`heartbeat_extended_20260909_0931.json`、`heartbeat_extended_20260909_0931_writeback.json`、固定`_snapshot.json`及原始核验`_ruler.json`。','']
p=B/'heartbeat_extended_20260909_0931.md';assert not p.exists();p.write_text('\n'.join(md))
print(json.dumps({'new_cells':2,'distinct_new_predictions':100,'writeback':cells,'pending_ruler':ruler_waiting,'outputs':str(B/'heartbeat_extended_20260909_0931.json')}),flush=True)
