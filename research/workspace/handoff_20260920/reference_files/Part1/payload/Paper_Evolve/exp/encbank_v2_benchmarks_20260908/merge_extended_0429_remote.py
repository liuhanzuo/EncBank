"""Merge two completed CPU audits; no model/tokenizer work."""
import json
from pathlib import Path
from datetime import datetime,timezone
B=Path('/data/liuhanzuo/encbank_v2_20260908/workspace/exp/encbank_v2_benchmarks_20260908')
def read(p):return json.loads(p.read_text())
d=read(B/'heartbeat_extended_20260909_0429.json');ib=read(B/'heartbeat_infinitebench_20260909_0429_new.json')
assert len(ib['writeback'])==3 and ib['checks']['official_rescores']==1053 and ib['checks']['new_source_token_pack_exact_input_own_cache_checks']==1053
old={(x['benchmark'],x['task'],x.get('length'),x['arm']) for hour in ('0129','0229','0329') for x in read(B/f'heartbeat_extended_20260909_{hour}.json')['writeback']}
for x in d['writeback']+ib['writeback']:assert (x['benchmark'],x['task'],x.get('length'),x['arm']) not in old and x['complete'] and x['n']==x['expected_n']
d['writeback']+=ib['writeback'];d['comparisons']+=ib['comparisons']
d['pending']=[p for p in d['pending'] if not(p['benchmark']=='infinitebench' and p['task']=='longbook_qa_eng')]
assert not any(p.get('coverage_complete') or any(c['coverage_complete'] for c in p.get('categories',[])) for p in d['pending'] if p['benchmark'] in ('locomo','infinitebench'))
d['checks']['all_new_full_IB_LoCoMo_groups_audited']=True;d['checks']['infinitebench']=ib['checks'];d['infinitebench_protocol']=ib['protocol']
d['infinitebench_detail_source']='heartbeat_infinitebench_20260909_0429_new.json'
d['narrative_review']={'body_change_required':False,'reason':'Existing mixed-results summary remains valid. New complete IB QA baselines are below V2, while V2 still trails full recompute on LongEval.',
    'limits':'No uniform cross-benchmark superiority; no significance claim; retain prior LongEval16k V2/fullKV82 tie and all low/zero new baseline scores.',
    'compact_writeback':'Fill existing LongEval rows. Extend existing RULER appendix baseline cells. Add three IB QA baseline scores to current appendix table or one adjacent sentence; MC baseline cells stay blank.'}
d['merged_at_utc']=datetime.now(timezone.utc).isoformat()
(B/'heartbeat_extended_20260909_0429.json').write_text(json.dumps(d,indent=2)+'\n')
out=['# 04:29 扩展基线准确率独立 CPU 核查','',
     '相对03:29及以前已核结果，仅处理新增完整基线单元。全部评分、源prompt重建、tokenizer、检索和SQLite检查在longjing-1 CPU执行，CUDA显式隐藏；未构造模型，未改论文、队列、同步器或任何GPU任务。V2/j0仅作已核参照，本轮未重评分或重新检查其generation cache。','',
     '## 本轮可写回完整格','',
     '| Benchmark / task | Length | Arm | n | Score (%) |','|---|---|---|---:|---:|']
for c in d['writeback']:out.append(f"| {c['benchmark']} / {c['task']} | {c.get('length','native')} | {c['arm']} | {c['n']} | {c['score_percent']:.2f} |")
out+=['','`pub`=原始Encbank；`pub_sink`=Encbank+BOS；`cbos`=isolated full KV控制，不能当作CacheBlend。实测0保留0；没有把未知或未完成项写为0。','',
      '## InfiniteBench English long-book QA：三条新完整基线','',
      '三臂均有4/4完成shard，索引精确覆盖0–350、官方ID唯一、每个receipt的new+reused与shard记录数匹配。每臂**351例**，不是264/351局部结果，也不是整个InfiniteBench suite。', '',
      '**1,053/1,053条官方answer F1重评分通过**，逐条与保存score一致，分片scores.json均值也一致；最终百分比分数是100乘以351例逐例F1均值，不能把它当整题正确率。三个新基线**1,053次source ID/answers、完整token、ordered pack、各自exact-input generation-cache命中**全部通过，缓存预测和n_tokens对应其本方法输出；从未要求不同方法预测相同。V2/j0只检查既有记录source/pack及共享参数配对，没有重评分或查其缓存。', '',
      '协议：stock Qwen3-8B BF16、SDPA及相同repeat_kv policy；pub/pub_sink有效j12，cbos有效j36。vendored官方yarn-mistral裸模板，无chat wrapper；显式context/query边界独立分词，无源上下文截断或padding；chunk512、BM25 top12、相同ordered selected pack；greedy cap40，首token EOS被抑制，此后可自然EOS。该accuracy路径不能当成固定长度serving测速。', '',
      f"本轮重建token范围：`{ib['protocol']['token_ranges']}`。全文token可能超过模型原始窗口，实际只读取所选chunk和完整query；不是把全文dense输入模型。", '',
      '相对已核V2/j0的描述性比较：','',
      '| New arm | Official F1 (%) | V2 F1, previous | j0 F1, previous | V2 − arm (pp) |','|---|---:|---:|---:|---:|']
for c in ib['comparisons']:out.append(f"| {c['new_arm']} | {c['score_percent']:.2f} | {c['v2_previously_verified_score_percent']:.2f} | {c['full_recompute_previously_verified_score_percent']:.2f} | {c['v2_minus_new_arm_pp']:+.2f} |")
out+=['','没有显著性检验，也不从该单任务推断整个InfiniteBench或跨任务绝对优势。MC基线未完整，继续空白。', '',
      '## LongEval与RULER新增格的检查','',
      f"LongEval新格共**{d['checks']['longeval_source_token_exact_input_cache_and_rescore_checks']}次**源label/expected/line count、完整token输入、各自generation参数缓存键和原评分检查通过。每格50例、receipt完整，源按原driver的crc32长度seed重新合成；V2/j0只用已核score及source metadata/共享参数对齐。", '',
      'LongEval是Encbank adaptation：random-letter label、six-digit value、completion prompt、首个至少四位数精确匹配，非官方LongChat模板/评分；stock模型、seed1234、chunk512、BM25 top12、cap16。**完整token与exact-input缓存检查不等于runtime selected-pack验证**；原日志没有存selected indices/完整runtime pack。', '',
      f"RULER新格共**{d['checks']['new_ruler_raw_rescores']}条**raw recall复算通过；completed/exit_code0、i=0–49、完整50例、保存summary一致；与已核V2/j0同格的task/length/i/answers/n_tokens和CLI（除arm/out）对齐，日志有PYTHONHASHSEED=0和准入记录。", '',
      'RULER沿用remote runtime/seed42、stock模型、top12/chunk512；五针是iter_bm25、iter_hop_topk4、实际cap60。**legacy RULER没有原始完整token、运行时selected pack或generation-cache key存证**，本轮仅raw评分+metadata/CLI/seed配对，不能升级为exact-pack运行时验证；five-needle不是多事实组合推理证明。', '',
      '## 尚未完整的基线任务/类别','',
      '| Benchmark / arm | 完成shard样本数 | 正式总n | Score |','|---|---:|---:|---|']
for p in d['pending']:
    if p['benchmark'] in ('infinitebench','locomo'):out.append(f"| {p['benchmark']} / {p['task']} / {p['arm']} | {p['n']} | {p['expected_n']} | — |")
out+=['','上述计数只来自完成shard，不包含活动shard的未完成行。LoCoMo正式各类别n为282/321/96/841/446、总计1986，JSON列出本轮各类覆盖；本轮仍没有新完整基线类别。未公布这些局部均值。', '',
      '## 紧凑写回与正文结论','',
      '1. Table16(a)保留现有行列，只填新增32k/64k格，不新增正文表。',
      '2. RULER新增五针8k及其它自然完成格补现有附录baseline网格；原4k及已核V2/j0保持，缺失格dash。',
      '3. InfiniteBench En.QA补三个完整baseline分数。若增加三列使Table16过宽，可保留现有V2/j0列并在邻接协议段用一句话给出Original/+BOS/isolated-full-KV的F1；不把MC部分shard分数填进来。',
      '4. 当前正文“相对full recompute，在InfiniteBench有增益，在LongEval/LoCoMo有缺口”的概括仍成立。本轮完整IB QA基线均低于V2，但不能扩大成全任务优越性；此前LongEval16k V2与isolated full KV同为82的平局、各新格低分/零分仍需保留。', '',
      '机器汇总：`heartbeat_extended_20260909_0429.json`；IB逐组验证：`heartbeat_infinitebench_20260909_0429_new.json`。复算入口：`audit_extended_0429_remote.py`、`audit_infinitebench_0429_remote.py`；两阶段完成后执行`merge_extended_0429_remote.py`。','']
(B/'heartbeat_extended_20260909_0429.md').write_text('\n'.join(out))
print(json.dumps({'writeback':d['writeback'],'checks':d['checks'],'output':str(B/'heartbeat_extended_20260909_0429.md')}))
