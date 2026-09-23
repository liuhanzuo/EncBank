"""Assemble only independently checked, fixed-snapshot new extended cells."""
import json
from pathlib import Path
from datetime import datetime,timezone
R=Path('/data/liuhanzuo/comem_v2_20260908');B=R/'workspace/exp/comem_v2_benchmarks_20260908'
def read(p):return json.loads(p.read_text())
s=read(B/'heartbeat_extended_20260909_1032_snapshot.json');r=read(B/'heartbeat_extended_20260909_1032_ruler.json');l=read(B/'heartbeat_locomo_20260909_1032_new.json')
assert s['timestamp_utc']==r['snapshot_timestamp_utc']==l['snapshot_timestamp_utc']
assert r['checks']['new_ruler_raw_rescores']==150 and r['checks']['old_outputs_rescored']==0
assert l['checks']['official_rescores']==l['checks']['new_exact_source_token_pack_own_generation_cache_checks']==3972
assert l['checks']['reconstructed_source_token_ordered_packs']==1986
assert l['checks']['reference_scorer_calls']==l['checks']['reference_cache_lookups']==0
assert len(r['writeback'])==3 and len(l['writeback'])==10
cells=r['writeback']+l['writeback'];assert sum(c['n'] for c in cells)==4122 and all(c['complete'] for c in cells)
ruler_paired=[]
for c in r['writeback']:
    rr=read(Path(c['source']))['rows'];a=c['arm'];targets={len(x['answers']) for x in rr};assert len(targets)==1
    per=targets.pop();matched=0
    for row in rr:
        v=row[a+'_recall']*per;assert abs(v-round(v))<1e-8;matched+=round(v)
    assert abs(100*matched/(50*per)-c['score_percent'])<1e-8
    c.update(raw_mean=c['score_percent']/100,exact_fraction={'matched_answer_targets':matched,'total_answer_targets':50*per,'targets_per_example':per,'mean_recall':matched/(50*per)})
    for ref,p in c['paired_contrasts'].items():ruler_paired.append({'benchmark':'ruler','task':c['task'],'length':c['length'],'new_arm':a,'reference_arm':ref,**p})
coverage=[];pending=[]
for item in r['pending']:
    if item['benchmark']!='locomo':continue
    c=dict(item)
    if c['arm'] in ('pub','pub_sink'):
        assert c['coverage_complete'] and c['n']==1986
        catcells=[x for x in l['writeback'] if x['arm']==c['arm']]
        assert len(catcells)==5 and sum(x['n'] for x in catcells)==1986
        c.update(writeback_eligible=True,independently_audited_complete_task=True,category_scores=[{'task':x['task'],'n':x['n'],'metric':x['metric'],'score_percent':x['score_percent']} for x in catcells],score_percent=None,mixed_category_mean_prohibited=True)
    else:
        assert c['arm']=='cbos' and c['n']==1788 and not c['coverage_complete']
        assert all(not x['coverage_complete'] and x['score_percent'] is None for x in c['categories']);pending.append(c)
    coverage.append(c)
assert len(coverage)==3 and len(pending)==1
waiting=[]
for name,j in s['state']['jobs'].items():
    if name.startswith(('ruler__pub__','ruler__pub_sink__','ruler__cbos__')) and name.endswith(('_128k','_256k')) and j.get('status')!='completed':
        waiting.append({'job':name,'status_at_snapshot':j.get('status'),'complete':False,'score_percent':None})
assert {x['job'] for x in waiting}=={'ruler__pub_sink__niah_multikey_1_256k','ruler__cbos__niah_multikey_1_256k'}
limits=[
 'RULER uses original CoMem adaptation answer-substring recall and legacy metadata/CLI/seed pairing only; exact full input tokens, selected runtime pack and generation-cache keys were not persisted.',
 'LoCoMo answerable categories1-4 use official F1 functions; category5 uses explicit option-decoding accuracy adaptation. Never mix the five categories into an overall mean.',
 'LoCoMo full chronological history including captions, official short-answer prompt, seed42 option mapping, Qwen chat template thinking disabled, greedy cap50/first-token EOS suppression; BM25 top12/chunk512 and no source truncation or padding.',
 'Every new LoCoMo source identity, full token sequence, ordered selected pack and own generation-cache prediction/n_tokens key was checked. Earlier V2/j0 scores were reused and source/pack matched without rescoring old predictions or opening their caches.',
 'Original LoCoMo official_qa_driver logs have no RULER native admission line: fixed-snapshot remote queue GPU2/3 job/exit records were checked, but historical per-job GPU admission memory readings were not persisted. These are accuracy results, not timing measurements.',
 'Isolated KV LoCoMo remains1788/1986, and all its categories remain incomplete; no partial mean is reported. Already audited MC, LongEval and older RULER cells are not rescored.',
 'Descriptive paired contrasts are not significance tests or causal explanations; remote accuracy establishes no local5090 time/memory advantage.'
]
negative=[
 'LoCoMo adversarial option accuracy reverses the answerable-category ranking: CoMem130/446=29.1479820628 and BOS255/446=57.1748878924 exceed V2 101/446=22.6457399103 and replay120/446=26.9058295964.',
 'BOS exceeds V2 on adversarial accuracy by34.5291479821pp (160 wins,280 ties,6 losses); original CoMem exceeds V2 by6.5022421525pp. No claim that V2 improves every LoCoMo category is supported.',
 'Both new baselines remain below V2 and replay in all four answerable categories. Metrics and generation adaptation differ from category5; the ranking reversal is retained without attributing a mechanism.',
 'At256k single-needle isolated KV100 ties V2/replay100 while BOS94 and original CoMem previously34. Original multi-key256 scores8 versus V2/replay92/100. No universal V2 advantage over isolated KV is supported.'
]
d={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'snapshot_timestamp_utc':s['timestamp_utc'],'snapshot_file':str(B/'heartbeat_extended_20260909_1032_snapshot.json'),
   'scope':'Only new complete extended baselines since09:33:50; fixed10:34:02 snapshot, no subsequent results included.',
   'execution':{'location':'remote CPU','torch_threads':2,'torch_interop_threads':16,'OMP_NUM_THREADS':2,'MKL_NUM_THREADS':2,'CUDA_VISIBLE_DEVICES':'','models_constructed':0,'cuda_initialized':False},
   'writeback':cells,'new_complete_cells':13,'new_predictions_audited':4122,'new_distinct_locomo_outputs':3972,
   'comparisons':l['comparisons'],'paired_counts':ruler_paired+l['paired_counts'],'locomo_reference_cells':l['reference_cells'],
   'coverage':coverage,'pending':pending,'pending_ruler_128k_256k':waiting,'checks':{'ruler':r['checks'],'locomo':l['checks']},
   'locomo_protocol':l['protocol'],'locomo_ranges':l['ranges'],'protocol_limits':limits,'negative_results':negative,
   'evidence_files':['heartbeat_extended_20260909_1032_ruler.json','heartbeat_locomo_20260909_1032_new.json','heartbeat_extended_20260909_1032_snapshot.json']}
for suffix,value in [('.json',d),('_writeback.json',{'schema_version':1,'snapshot_timestamp_utc':s['timestamp_utc'],'new_complete_cells':cells,'new_complete_cell_count':13,'new_predictions_audited':4122,'new_distinct_locomo_outputs':3972,'comparisons':l['comparisons'],'paired_counts':d['paired_counts'],'pending':pending,'pending_ruler_128k_256k':waiting,'protocol_limits':limits,'negative_results':negative,'evidence':str(B/'heartbeat_extended_20260909_1032.json')})]:
    p=B/('heartbeat_extended_20260909_1032'+suffix);assert not p.exists();p.write_text(json.dumps(value,indent=2)+'\n')
md=['# 10:32 扩展基线独立核验','',f"固定实际远端快照：{s['timestamp_utc']} UTC（北京时间10:34:02），main306/374。只纳入09:31之后、该快照前完整的新单元，不追收之后的结果。",'',
 '本轮新增13格、4122个不同新预测：RULER三格150例；LoCoMo原版和+BOS两整臂各1986例，拆成10个类别格，总计3972个不同输出。没有将每类别误计为完整1986例。','',
 '全部重评分、tokenization、检索pack与SQLite检查在远端CPU执行，Torch2线程/interop16、OMP2/MKL2、CUDA隐藏且未初始化，零模型构造。未修改论文、根文档、队列或GPU任务。旧MC、LongEval、RULER不重评分；LoCoMo旧V2/j0仅复用已核分数和配对身份，不调用旧评分器或旧cache。','',
 '## 新RULER三格','',
 '| Task | Length | New baseline | n | Score (%) |','|---|---|---|---:|---:|']
for c in r['writeback']:md.append(f"| {c['task']} | {c['length']} | {c['arm']} | {c['n']} | {c['score_percent']:.2f} |")
md+=['','150条raw通过原CoMem RULER substring-recall复算、连续index0–49、完整summary、无OOM、CLI/seed和准入日志。与旧参考task/length/i/answers/n_tokens一致。单针BOS47/50=94%，isolated50/50=100%；多键原版均值8%。','',
 'stock Qwen3-8B、j12、top12/chunk512、seed42、BM25、greedy cap48。历史raw缺完整tokens、runtime selected pack和generation-cache键，只能声明同runtime metadata/CLI/seed配对；不能升级为exact-token/pack验证。256k单针isolated与V2/Replay持平，不能声称V2在每个任务都更好。','',
 '## LoCoMo完整类别','',
 '| Category | n | CoMem | +BOS | V2 (prior) | Replay (prior) |','|---|---:|---:|---:|---:|---:|']
matrix={(c['task'],c['arm']):c for c in l['writeback']+l['reference_cells']}
for cat,n in {1:282,2:321,3:96,4:841,5:446}.items():
    t=f'category_{cat}';md.append(f"| {cat} | {n} | "+' | '.join(f"{matrix[t,a]['score_percent']:.10f}" for a in ('pub','pub_sink','fix_all','j0'))+' |')
md+=['','类别1–4分别为multi-hop、temporal、open-domain、single-hop，使用官方F1；类别5为adversarial，使用明确记录的option-decoding accuracy适配，不能混成一个总均值。','',
 '两臂共20个完整分片，逐个核对completed/exit0、1986个唯一conv/qa源ID、连续索引、正式类别数、原始答案/evidence、seed42的选项映射和expected配置。3972次新官方评分及scored_prediction一致，1986次完整源tokens/有序BM25 pack重建，3972次自身SQLite精确generation-key/prediction/n_tokens检查全部通过；每个源同时与旧V2/j0的同源样本和pack匹配。','',
 '完整源13,539–25,686 tokens，read pack5,738–6,252 tokens；完整时间顺序会话及图片caption、原HF短答prompt、Qwen chat模板且thinking关闭、完整query无截断/padding。固定cap50、greedy、首步EOS抑制后自然EOS；原版j12且不写sink，+BOS j12写sink，统一repeat_kv SDPA且禁GQA。它是CoMem检索生成适配，不是未经修改的官方HF生成路径。','',
 '原official_qa_driver日志没有RULER的native admission行。核验的是实际快照内remote GPU2/3的job/PID/exit记录；逐任务准入显存读数未被历史日志持久化，因此不声称这类额外证据，也不把远端准确率输出用于系统成本结论。','',
 '## 与已核参考的均值差','',
 '下表方向为V2或Replay减去新基线，负数表示新基线更高；只作描述性比较。','',
 '| Category | Baseline | V2 minus baseline (pp) | Replay minus baseline (pp) |','|---|---|---:|---:|']
for c in l['comparisons']:md.append(f"| {c['task']} | {c['new_arm']} | {c['v2_minus_new_arm_pp']:+.10f} | {c['j0_minus_new_arm_pp']:+.10f} |")
md+=['','**必须保留的负结果：** 四个可回答类别中V2和Replay均高于新两基线；对抗类别则相反：CoMem130/446=29.15%，+BOS255/446=57.17%，V2旧101/446=22.65%，Replay旧120/446=26.91%。+BOS对V2的逐例胜/平/负为160/280/6，原版为72/331/43。新BOS提高了对抗选项准确率这一事实不能被混合总分隐藏，也不能据此直接推出其机制或一般性拒答校准更好。全部20个LoCoMo描述性配对对照均保留在机器JSON，未做显著性检验。','',
 '## 尚缺完整结果及写回建议','',
 'isolated KV LoCoMo仍9/10分片、1788/1986；五类覆盖260/282、284/321、86/96、754/841、404/446，全部未齐且分数null。RULER只剩BOS/isolated的multi-key256k两格未完整。MC五方法和LongEval此前已核完整，本轮不重审。','',
 '既有附录LoCoMo块可扩为Category/n/CoMem/+BOS/Isolated KV/V2/Replay七列，在caption说明前四类F1和对抗option accuracy；KV继续空白。不增正文表。RULER仅补本轮三格。正文/附录需明确LoCoMo改进限于四个可回答类别，对抗类别的baseline反超单独写出。','',
 '交付：`heartbeat_extended_20260909_1032.md/.json`、`_writeback.json`、固定`_snapshot.json`、`_ruler.json`和`heartbeat_locomo_20260909_1032_new.json`。','']
p=B/'heartbeat_extended_20260909_1032.md';assert not p.exists();p.write_text('\n'.join(md))
print(json.dumps({'new_complete_cells':13,'new_distinct_predictions':4122,'locomo_distinct':3972,'waiting_ruler':waiting,'pending_locomo':pending,'output':str(B/'heartbeat_extended_20260909_1032.json')}),flush=True)
