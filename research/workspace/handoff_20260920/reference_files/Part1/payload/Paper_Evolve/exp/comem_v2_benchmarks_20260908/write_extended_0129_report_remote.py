"""Finalize the bounded CPU audit report without model computation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
R=Path('/data/liuhanzuo/comem_v2_20260908')
B=R/'workspace/exp/comem_v2_benchmarks_20260908'
def read(p):return json.loads(p.read_text())
p=B/'heartbeat_extended_20260909_0129.json';d=read(p)
state=read(R/'outputs/queue_v2/full/state.json')
task='niah_single_2';length='4k';arm='pub';jobid=f'ruler__{arm}__{task}_{length}'
job=state['jobs'].get(jobid,{})
if job.get('status')=='completed' and job.get('exit_code')==0:
    path=R/f'outputs/benchmarks_v2/full/ruler/{arm}/{task}_{length}.json';obj=read(path);rows=obj['rows']
    assert len(rows)==50 and [r['i'] for r in rows]==list(range(50)) and obj['arms']==[arm]
    score=100*sum(sum(a.lower() in r['pub_out'].lower() for a in r['answers'])/len(r['answers']) for r in rows)/50
    assert abs(score-obj['summary'][f'{task}/{length}'][arm])<1e-9 and '[remote admission]' in Path(job['log']).read_text()
    for r in rows:
        val=sum(a.lower() in r['pub_out'].lower() for a in r['answers'])/len(r['answers'])
        assert abs(val-r['pub_recall'])<1e-9 and r['pub_out']!='[OOM]'
    for refarm in ('fix_all','j0'):
        ref=read(R/f'outputs/benchmarks_v2/full/ruler/{refarm}/{task}_{length}.json')
        assert {k:v for k,v in obj['args'].items() if k not in ('arms','out')}=={k:v for k,v in ref['args'].items() if k not in ('arms','out')}
        for a,b in zip(rows,ref['rows']):assert all(a[k]==b[k] for k in ('task','length','i','answers','n_tokens'))
    d['writeback'].append({'benchmark':'ruler','task':task,'length':length,'arm':arm,'n':50,'expected_n':50,'complete':True,'metric':'legacy answer substring recall',
        'score_percent':score,'display':f'{score:.2f}','sources':[str(path)],'source_metadata_pairing':True,'exact_pack_claim_allowed':False})
else:d['pending'].append({'benchmark':'ruler','task':task,'length':length,'arm':arm,'complete':False,'score_percent':None,'remote_status':job.get('status','missing')})
d['final_inventory_timestamp_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
p.write_text(json.dumps(d,indent=2)+'\n')
out=['# 01:29 扩展准确率独立 CPU 核查','',
     '仅审 InfiniteBench、CoMem-adapted LongEval 与 RULER；LongBench、LoCoMo、BABILong 由其他代理负责。本轮没有 GPU/模型计算，没有改论文、同步器或实验结果。全部重新分词、检索、SQLite 和评分计算在 longjing-1 CPU 执行，CUDA 显式隐藏；不使用远端耗时作为论文测速。','',
     '## 新完整结果格','',
     '| Task | Length | Method | n | Score (%) |','|---|---|---|---:|---:|']
for c in d['writeback']:
    out.append(f"| {c['benchmark']} / {c['task']} | {c.get('length','native')} | {c['arm']} | {c['n']} | {c['score_percent']:.2f} |")
mc=next(c for c in d['comparisons'] if c['task']=='longbook_choice_eng')
out+=['',f"InfiniteBench English long-book choice：V2 **130/229 = 56.77%**，全层重算 j0 **121/229 = 52.84%**，差 **{mc['v2_minus_j0_pp']:+.2f} pp**。逐例正确性列联计数为 `{mc['paired_counts']}`，仅报告描述性均值，未进行显著性检验。",'',
      'RULER multi-key 256k：V2 **92.00%**、j0 **100.00%**，差 **−8.00 pp**。这项负结果必须保留，不能写成各扩展任务均无损。多键任务在四个候选 key 中查询一个答案，不是要求同时组合四条事实。','',
      '## InfiniteBench 229 例配对与官方评分证据','',
      '两臂各 4 个已完成 shard，源索引精确覆盖 0–228，ID 唯一，无缺失或重复；每个 receipt 的 new+reused generation 数与对应 shard 记录数相等。两臂除 arm 外 driver_options 一致，reader 实际为 CoMemLower j12 与 CoMem j0，冻结 Qwen3-8B BF16、SDPA 和 repeat_kv kernel policy 一致。',
      '',
      '逐条从原始官方 longbook_choice_eng.jsonl 构造 vendored yarn-mistral 模板；使用 Qwen tokenizer 重新编码全文与完整 query，在显式边界独立分词，再以 BM25 top12/chunk512 重建 ordered pack。458/458 个 source ID、accepted answers、input token 数与整个 pack 字典通过。每个精确 token+generation 参数 SHA256 键都在该方法/分片自己的只读 generations.sqlite3 中命中，预测字符串与 n_tokens 也相同；未读取缓存 elapsed_s。',
      '',
      '458/458 条预测用 vendored 官方 `get_score_one_longbook_choice_eng` 重新评分，逐条与保存 score 一致，分片均值与 scores.json 一致。称为官方 option accuracy；该 scorer 含选项字母及答案文本提取/容错规则，不应改写成自行定义的首字母准确率。',
      '',
      '无截断、padding 或 chat wrapper；裸 yarn-mistral prompt，不能套用 LongBench 的 thinking-disabled chat 描述。greedy 生成 cap40，首 token EOS 被抑制，此后可自然 EOS。完整 source 可超过模型原始位置上限，但读取只用所选 chunk 和完整 query；这不是直接 dense full-source 长窗推理。',
      '',f"重新计算的 token 范围：`{mc['token_ranges']}`。",'',
      '## Table16 漏列的 RULER 单元','',
      '| Task | Length | V2 (%) | j0 (%) | n per arm | V2 − j0 (pp) |','|---|---|---:|---:|---:|---:|']
for c in d['ruler_matrix']:
    out.append(f"| {c['task']} | {c['length']} | {c['scores_percent']['fix_all']:.2f} | {c['scores_percent']['j0']:.2f} | 50 | {c['delta_v2_minus_j0_pp']:+.2f} |")
out+=['',
      '本轮对这 10 个长度/任务配对的 1,000 条 raw 预测做了轻量 recall 重评分；不是重新运行模型。128k five-needle 已在 `heartbeat_accuracy_20260908_2304.md` 独立检查过，single128k 在 `audit_additional_2050.py` 对应结果中检查过；本轮仅把它们并入紧凑表的数字核对。',
      '',
      '每臂均有 completed/exit_code0，50 条连续 i=0–49，逐条 task/length/i/answers/n_tokens 配对相同，CLI 除 arm/out 外一致，两份日志都有 PYTHONHASHSEED=0 和远端准入记录。统一 seed42、stock Qwen3-8B、j12/top12/chunk512；single/multi 的 auto 实为 BM25、cap48；variable_tracking 的 auto 实为 iter_bm25、iter_hop_topk4、实际 cap60。后者仍按论文中的 five-needle retrieval 命名，不据此宣称事实链推理。',
      '',
      '**证据边界：legacy RULER raw 没有持久化完整 token IDs、运行时 selected pack 或 generation-cache key。** 本轮确认 metadata/答案/长度/CLI/seed 配对，未从源码重建 256k 全文并声称它就是已运行输入，也不能称为 exact-pack runtime verification。recall 使用 CoMem RULER 适配路径的 `_string_match_all_one`，不冒称完整官方 RULER suite。不要把旧 Windows Python/hash/encoding 的 16k 合成样本当作这组远端完全相同样本；需要精确 16k follow-up 时引用既有 matched16k 结果。',
      '',
      '## 新方法与未知项','']
for c in d['comparisons']:
    if c['task']=='longeval lines/4k':
        out.append(f"LongEval pub4k 已完整 50 例，分数 `{c['scores_percent']}`。本轮重新合成 50 条 prompt，三方法共150个源 label/expected/line count、token输入和完整 generation参数缓存键逐条对齐；只重评分新方法及配对参照。历史 raw 没有记录 selected indices，因此不声称另有原始 pack 日志。该任务为 CoMem 适配，首个至少4位数的精确值评分，非官方 LongChat 协议。")
out+=['','未完整任务的表格分数保持空白：','', '| Task | Arm | 已完成 shard 样本数 | Expected n | Score |','|---|---|---:|---:|---|']
for c in d['pending']:
    if c['benchmark']=='infinitebench':out.append(f"| {c['task']} | {c['arm']} | {c['n_completed_shards']} | {c['expected_n']} | — |")
out+=['','这里的已完成 shard 数不包括活动 shard 的未提交进度；不能以 88/351 的 pub QA 局部均值填完整 benchmark。其他方法/长度未知不是0。','',
      '## Table16 紧凑补齐建议','',
      '1. 保留 LongEval 六长度两行，若希望展示新 baseline，可加 pub 一行：4k有值，其余 dash；不需要为单格加宽全表。',
      '2. InfiniteBench 放两行：En.QA / Answer F1 / n351 与 En.MC / Option accuracy / n229；caption 改为两个 English long-book tasks，不写已覆盖完整 InfiniteBench。',
      '3. 把零散 RULER 行合成 task×method 的六行长度矩阵，优先列4k/128k/256k。五针256k未计划/未测留dash；8k/64k五针的92.0/97.6（j0均100）可用紧邻短句补齐，或在宽度允许时加8k/64k列并引用现有主表单/多针列，保持旧/新协议边界。',
      '4. BABILong 和 LoCoMo 的最终布局由负责代理补齐，本报告不替其分数作结论。',
      '',
      '机器 JSON：`heartbeat_extended_20260909_0129.json`。可复算入口：`audit_extended_0129_remote.py` 与 `write_extended_0129_report_remote.py`。JSON 每个结果保留实际 remote source 路径、配对强度与空值；本轮只新增报告和CPU审计脚本。','']
(B/'heartbeat_extended_20260909_0129.md').write_text('\n'.join(out))
print(json.dumps({'writeback':d['writeback'],'ruler_matrix':d['ruler_matrix'],'report':str(B/'heartbeat_extended_20260909_0129.md')},ensure_ascii=False))
