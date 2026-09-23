"""Final remaining baseline cells, fixed11:34:38 snapshot; no model or old rescore."""
import json
from pathlib import Path
from datetime import datetime,timezone
R=Path('/data/liuhanzuo/encbank_v2_20260908');B=R/'workspace/exp/encbank_v2_benchmarks_20260908'
def read(p):return json.loads(p.read_text())
s=read(B/'heartbeat_extended_20260909_1132_snapshot.json');r=read(B/'heartbeat_extended_20260909_1132_ruler.json');l=read(B/'heartbeat_locomo_20260909_1132_new.json');old=read(B/'heartbeat_locomo_20260909_1032_new.json')
assert s['timestamp_utc']==r['snapshot_timestamp_utc']==l['snapshot_timestamp_utc']
assert r['checks']['new_ruler_raw_rescores']==100 and r['checks']['old_outputs_rescored']==0
assert l['checks']['official_rescores']==l['checks']['new_exact_source_token_pack_own_generation_cache_checks']==l['checks']['reconstructed_source_token_ordered_packs']==1986
assert l['checks']['reference_scorer_calls']==l['checks']['reference_cache_lookups']==0
assert len(r['writeback'])==2 and len(l['writeback'])==5
cells=r['writeback']+l['writeback'];assert sum(c['n'] for c in cells)==2086 and all(c['complete'] for c in cells)
ruler_paired=[];raw={}
for c in r['writeback']:
    assert c['task']=='niah_multikey_1' and c['length']=='256k' and c['n']==50
    rr=read(Path(c['source']))['rows'];a=c['arm'];raw[a]=rr;counts={len(x['answers']) for x in rr};assert len(counts)==1
    targets=counts.pop();matched=0
    for row in rr:
        v=row[a+'_recall']*targets;assert abs(v-round(v))<1e-8;matched+=round(v)
    assert abs(100*matched/(50*targets)-c['score_percent'])<1e-8
    c.update(raw_mean=c['score_percent']/100,exact_fraction={'matched_answer_targets':matched,'total_answer_targets':50*targets,'targets_per_example':targets,'mean_recall':matched/(50*targets)})
    for ref,p in c['paired_contrasts'].items():ruler_paired.append({'benchmark':'ruler','task':c['task'],'length':c['length'],'new_arm':a,'reference_arm':ref,**p})
assert set(raw)=={'cbos','pub_sink'}
diff=[]
for a,b in zip(raw['cbos'],raw['pub_sink']):
    assert all(a[k]==b[k] for k in ('task','length','i','answers','n_tokens'))
    diff.append(a['cbos_recall']-b['pub_sink_recall'])
bos_pair={'benchmark':'ruler','task':'niah_multikey_1','length':'256k','new_arm':'cbos','reference_arm':'pub_sink','paired_n':50,
          'new_arm_wins':sum(x>1e-12 for x in diff),'ties':sum(abs(x)<=1e-12 for x in diff),'new_arm_losses':sum(x< -1e-12 for x in diff),
          'new_arm_minus_reference_pp':100*sum(diff)/50,'direction':'isolated KV relative to BOS, both new and independently rescored this round',
          'sample_metadata_CLI_seed_pairing_only':True,'exact_input_pack_claim':False,'significance_test_performed':False}
assert abs(bos_pair['new_arm_minus_reference_pp']+24)<1e-8;ruler_paired.append(bos_pair)
matrix={(c['task'],c['arm']):c for c in old['writeback']+old['reference_cells']+l['writeback']}
assert len(matrix)==25
fullmatrix=[]
for cat,n in {1:282,2:321,3:96,4:841,5:446}.items():
    task=f'category_{cat}';row={'task':task,'category':cat,'n':n,'metric':'adversarial option accuracy' if cat==5 else 'official F1','arms':{}}
    for arm in ('pub','pub_sink','cbos','fix_all','j0'):
        c=matrix[task,arm];assert c['n']==n
        row['arms'][arm]={'score_percent':c['score_percent'],'display':f"{c['score_percent']:.2f}",'rescored_this_round':arm=='cbos'}
        if 'correct' in c:row['arms'][arm]['correct']=c['correct']
    fullmatrix.append(row)
coverage=[]
for c in r['pending']:
    if c['benchmark']!='locomo':continue
    assert c['coverage_complete'] and c['n']==1986 and all(x['coverage_complete'] for x in c['categories'])
    c=dict(c);c.update(writeback_eligible=True,independently_audited_complete_task=True,score_percent=None,mixed_category_mean_prohibited=True,category_scores=[{'task':f'category_{cat}','n':n,'score_percent':matrix[f'category_{cat}',c['arm']]['score_percent']} for cat,n in {1:282,2:321,3:96,4:841,5:446}.items()])
    coverage.append(c)
assert len(coverage)==3
extjobs={name:j for name,j in s['state']['jobs'].items() if name.split('__')[0] in ('ruler','longeval','infinitebench','locomo') and name.split('__')[1] in ('pub','pub_sink','cbos')}
remaining=[{'job':name,'status':j.get('status'),'score_percent':None} for name,j in extjobs.items() if j.get('status')!='completed' or j.get('exit_code')!=0]
assert not remaining
limits=[
 'RULER original Encbank adaptation answer-substring recall; legacy metadata/CLI/seed pairing only. Exact full input tokens, selected runtime pack and generation-cache keys were not persisted.',
 'LoCoMo answerable categories1-4 official F1; category5 explicit option-decoding accuracy adaptation. No combined five-category average.',
 'Whole chronological conversations and captions; official short-answer prompt, stable seed42 option mapping, Qwen chat template with thinking disabled; greedy cap50, first-token EOS suppression followed by natural EOS; BM25 top12/chunk512, no source truncation/padding.',
 'All1986 new isolated-KV source identities, full input tokens, ordered selected packs and own SQLite generation keys/predictions/n_tokens checked. Old V2/j0/BOS references use previously verified scores and source/pack metadata without old scorer or cache calls; BOS aggregate identity checked against1032 report.',
 'Historical QA job completion/GPU2/3/PID/exit records checked. Official QA logs do not persist per-job admission memory samples or RULER native admission lines; no such evidence or system-performance conclusion is invented.',
 'Descriptive paired contrasts are not significance tests. Larger cached state and repaired low-level visibility do not imply uniformly better quality or abstention behavior.',
 'All prior MC/LongEval/RULER/LoCoMo scores are reused from independent reports, not rescored this round; all new results are bound to this fixed snapshot.'
]
negative=[
 'At256k multi-key BOS62 exceeds isolated KV38 by24pp despite the larger isolated cache; prior original8/V2 92/replay100 unchanged.',
 'LoCoMo open-domain isolated KV18.2683881893 exceeds V2 14.0170630406 by4.2513251486pp and replay16.1305182609 by2.1378699284pp.',
 'Open-domain isolated KV versus V2 has22 wins/57 ties/17 losses; versus replay17 wins/52 ties/27 losses. Higher mean F1 does not imply winning on more examples, because magnitudes differ.',
 'LoCoMo adversarial isolated KV109/446=24.4394618834 exceeds V2 101/446=22.6457399103 by1.7937219731pp, but is below replay120/446=26.9058295964 and BOS255/446=57.1748878924.',
 'V2 exceeds isolated KV only in multi-hop, temporal and single-hop among these five categories. Previous V2 improvements over pub/BOS in four answerable categories must not be extended to all five methods.'
]
d={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'snapshot_timestamp_utc':s['timestamp_utc'],'snapshot_file':str(B/'heartbeat_extended_20260909_1132_snapshot.json'),
   'scope':'Only new complete extended baselines since10:34:02, fixed11:34:38 snapshot; no later outputs included.',
   'execution':{'location':'remote CPU','torch_threads':2,'torch_interop_threads':16,'OMP_NUM_THREADS':2,'MKL_NUM_THREADS':2,'CUDA_VISIBLE_DEVICES':'','models_constructed':0,'cuda_initialized':False},
   'writeback':cells,'new_complete_cells':7,'new_predictions_audited':2086,'new_distinct_locomo_outputs':1986,
   'comparisons':l['comparisons'],'paired_counts':ruler_paired+l['paired_counts'],'locomo_reference_cells':l['reference_cells'],
   'complete_locomo_five_method_matrix':fullmatrix,'coverage':coverage,'pending':[],'pending_ruler_128k_256k':[],
   'baseline_extended_queue_completeness':{'jobs':len(extjobs),'complete':len(extjobs),'remaining':remaining,'boundary':'Queue completion only; old raw-score verification inherited from prior independent reports.'},
   'checks':{'ruler':r['checks'],'locomo':l['checks']},'locomo_protocol':l['protocol'],'locomo_ranges':l['ranges'],'protocol_limits':limits,'negative_results':negative,
   'evidence_files':['heartbeat_extended_20260909_1132_ruler.json','heartbeat_locomo_20260909_1132_new.json','heartbeat_extended_20260909_1132_snapshot.json']}
for suffix,value in [('.json',d),('_writeback.json',{'schema_version':1,'snapshot_timestamp_utc':s['timestamp_utc'],'new_complete_cells':cells,'new_complete_cell_count':7,'new_predictions_audited':2086,'new_distinct_locomo_outputs':1986,'comparisons':l['comparisons'],'paired_counts':d['paired_counts'],'complete_locomo_five_method_matrix':fullmatrix,'pending':[],'pending_ruler_128k_256k':[],'protocol_limits':limits,'negative_results':negative,'evidence':str(B/'heartbeat_extended_20260909_1132.json')})]:
    p=B/('heartbeat_extended_20260909_1132'+suffix);assert not p.exists();p.write_text(json.dumps(value,indent=2)+'\n')
md=['# 11:32 扩展基础基线收尾核验','',f"固定实际快照：{s['timestamp_utc']} UTC（北京时间11:34:38），main333/374。只纳入10:32后、该快照前新完整单元。",'',
 '本轮7格、2086个不同新预测：RULER两格各50；isolated KV LoCoMo1986例，按正式类别拆为5格。所有重评分、tokenization、有序pack和generation-cache检查均在远端CPU完成，Torch2/interop16、OMP2/MKL2、CUDA隐藏且未初始化，零模型构造。没有修改论文、根文档、队列或GPU任务。','',
 '## RULER最后两格','',
 '| Task | Length | +BOS | Isolated KV | n per arm |','|---|---|---:|---:|---:|','| Multi-key | 256k | 62.00 | 38.00 | 50 |','',
 '原版8/V2 92/Replay100沿用旧已核记录。新100条raw通过原Encbank substring-recall、连续索引、完整summary、无OOM、CLI/seed和native准入日志核验。stock Qwen3-8B、j12、top12/chunk512、BM25、seed42、greedy cap48。历史没有完整输入tokens、runtime selected pack或generation-cache存证，所有RULER配对仅限同runtime metadata/CLI/seed，不升级为exact-token/pack。','',
 f"isolated KV比+BOS低24点，其逐例胜/平/负为{bos_pair['new_arm_wins']}/{bos_pair['ties']}/{bos_pair['new_arm_losses']}（n50）。保留较大缓存反而更低的结果。",'',
 '## LoCoMo五方法完整矩阵','',
 '| Category | n | Encbank | +BOS | Isolated KV | V2 | Replay |','|---|---:|---:|---:|---:|---:|---:|']
for row in fullmatrix:md.append(f"| {row['category']} | {row['n']} | "+' | '.join(f"{row['arms'][a]['score_percent']:.10f}" for a in ('pub','pub_sink','cbos','fix_all','j0'))+' |')
md+=['','类别1–4依次为multi-hop、temporal、open-domain、single-hop，使用官方F1；类别5为adversarial，采用明确记录的robust option-decoding accuracy适配，不能计算混合类别总分。只有KV列为本轮新评分，其他四列沿用已核报告。','',
 'KV十个完整分片恰覆盖1986个唯一conv/qa源ID和连续全局索引；正式类别分母282/321/96/841/446。1986个新raw按官方类别函数及明确选项解码规则复算，与保存score/scored_prediction一致；1986次完整源tokenization、有序BM25 selected pack和自身SQLite精确generation key/prediction/n_tokens检查全部通过。每个源pack与V2/j0/BOS参考一致，整组输入key摘要也与上轮pub/BOS审计一致。旧V2/j0逐记录分数身份、BOS旧类别均值身份已核对，没有旧评分器调用或旧cache查询。','',
 '完整源13,539–25,686 tokens，read pack5,738–6,252 tokens。完整时间顺序对话与caption、官方短答prompt、seed42选项映射、Qwen chat模板关闭thinking、完整query无截断/padding；greedy cap50、首步EOS抑制后自然EOS。isolated KV为EncbankLower有效j36、写sink、stock BF16/SDPA、统一repeat_kv禁GQA，是Encbank适配而非未经修改的HF生成或CacheBlend。','',
 '沿上轮已校正的历史记录口径：核实际GPU2/3 job/PID/exit0和完整receipt；原QA日志没有RULER native admission行，也未持久化每任务准入显存读数。不能虚构该证据，更不能据远端准确率推断本地5090系统成本。','',
 '## 均值差及必须保留的反例','',
 '方向为参考减去isolated KV，负数表示KV更高。','',
 '| Category | V2 minus KV (pp) | Replay minus KV (pp) | BOS minus KV (pp) |','|---|---:|---:|---:|']
for c in l['comparisons']:md.append(f"| {c['task']} | {c['v2_minus_new_arm_pp']:+.10f} | {c['j0_minus_new_arm_pp']:+.10f} | {c['bos_minus_new_arm_pp']:+.10f} |")
md+=['','**新的反例：** open-domain KV18.27高于V2 14.02和Replay16.13；adversarial KV24.44高于V2 22.65，但低于Replay26.91及BOS57.17。V2对KV的改进仅在multi-hop/temporal/single-hop三类，不可沿用“V2优于所有可回答类别基线”的概括。此前V2对pub/BOS四个可回答类别的提升仍成立，必须明确比较对象。','',
 'open-domain KV相对V2胜/平/负22/57/17，相对Replay17/52/27；虽然相对Replay均值更高，但赢的样本更少，反映F1差值幅度不同，不能把均值优势说成逐例多数优势。adversarial KV为109/446，较V2多8例，相对V2胜/平/负26/402/18；相对BOS为10/280/156。全15个LoCoMo描述性配对比较保存在JSON，不声称显著性或解释成一般性校准机制。','',
 '## 收尾与写回','',
 f"固定快照中RULER/LongEval/InfiniteBench/LoCoMo三个基础baseline的{len(extjobs)}个计划job均completed/exit0；这仅是queue完整性，旧评分核验继承此前独立报告。本轮范围已无未完整RULER/LoCoMo格，不把主队列其他任务也宣布完成。",'',
 '既有附录只填Multi-key256k的BOS62/KV38及LoCoMo KV五类21.47/18.45/18.27/47.20/24.44。不增加正文表。相关概括必须同时保留open-domain KV反超和对抗类别BOS领先；不能用混合总分或更大cache容量掩盖反例。','',
 '文件：`heartbeat_extended_20260909_1132.md/.json/_writeback.json/_snapshot.json/_ruler.json`和`heartbeat_locomo_20260909_1132_new.json`。','']
p=B/'heartbeat_extended_20260909_1132.md';assert not p.exists();p.write_text('\n'.join(md))
print(json.dumps({'new_complete_cells':7,'new_distinct_predictions':2086,'locomo_distinct':1986,'baseline_jobs_complete':len(extjobs),'ruler_kv_vs_bos':bos_pair,'output':str(B/'heartbeat_extended_20260909_1132.json')}),flush=True)
