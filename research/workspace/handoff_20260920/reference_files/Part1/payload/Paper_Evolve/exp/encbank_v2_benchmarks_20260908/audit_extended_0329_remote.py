"""New extended baseline accuracy cells only; remote CPU, no models."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import sys,json,sqlite3,hashlib,random,zlib
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
R=Path('/data/liuhanzuo/encbank_v2_20260908');B=R/'workspace/exp/encbank_v2_benchmarks_20260908';O=R/'outputs/benchmarks_v2/full'
sys.path[:0]=[str(B),str(R/'workspace/Encbank'),str(R/'workspace/exp')]
from transformers import AutoTokenizer
from eval import longeval as le
from eval import ruler as ru
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def near(a,b):assert abs(a-b)<1e-9,(a,b)
def native(run):
    m=read(run/'COMPLETED.json');assert m['status']=='completed'
    return Path(m['output_dir']),m,read(run/'run_config.json')
def key_for(ids,options):return hashlib.sha256(json.dumps({'tokens':ids.tolist(),'generation':options},sort_keys=True,separators=(',',':')).encode()).hexdigest()
prior={'writeback':read(B/'heartbeat_extended_20260909_0129.json')['writeback']+read(B/'heartbeat_extended_20260909_0229.json')['writeback']}
known={(c['benchmark'],c['task'],c.get('length'),c['arm']) for c in prior['writeback']}
state=read(R/'outputs/queue_v2/full/state.json')
d={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'execution':'remote CPU; CUDA hidden; no model construction or GPU tasks',
   'writeback':[],'comparisons':[],'pending':[],'checks':{},'prior_reference':['heartbeat_extended_20260909_0129.json','heartbeat_extended_20260909_0229.json']}
tok=AutoTokenizer.from_pretrained(R/'models/Qwen3-8B',local_files_only=True)
new_le=[]
for arm in ('pub','pub_sink','cbos'):
    for length in ('4k','8k','16k','32k','64k','128k'):
        run=O/'longeval'/arm/length
        if not (run/'COMPLETED.json').exists():
            d['pending'].append({'benchmark':'longeval','task':'lines','length':length,'arm':arm,'complete':False,'score_percent':None});continue
        if ('longeval','lines',length,arm) not in known:new_le.append((arm,length))
le_checks=0
for length in sorted({length for arm,length in new_le},key=lambda s:int(s[:-1])):
    targets=[arm for arm,l in new_le if l==length];allarms=targets;objects={};dbs={};options=None;origins={}
    for arm in allarms:
        run=O/'longeval'/arm/length;out,m,c=native(run);o=c['driver_options'];obj=read(out/f'longeval_{length}.json')
        assert m['new_generations']+m['reused_generations']==50 and len(obj['records'])==50
        assert [r['sample_index'] for r in obj['records']]==list(range(50))
        assert c['scoring']=='unchanged legacy driver; do not relabel as official'
        assert o['model_path']==str(R/'models/Qwen3-8B') and o['dtype']=='bfloat16' and o['attn_impl']=='sdpa'
        assert o['resume_j']==12 and o['selector']=='bm25' and o['topk']==12 and o['chunk_size']==512 and o['sink_tokens']=='bos'
        assert o['seed']==1234 and o['max_new_tokens']==16 and o['lengths']==[length] and o['num_samples']==50 and o['baseline']=='none' and o['lora_adapter']==''
        if options is None:options=o
        assert o==options
        ri=m['reader'];assert ri['arm']==arm and ri['effective_j']==(36 if arm=='cbos' else 0 if arm=='j0' else 12)
        assert ri['class']==('EncbankLower' if arm in ('fix_all','cbos') else 'Encbank')
        assert ri['write_sink']==(arm in ('fix_all','cbos','pub_sink'))
        assert ri['kernel_policy']=='s15 repeat_kv: use_gqa_in_sdpa=False (all arms)'
        objects[arm]=obj;dbs[arm]=sqlite3.connect(f'file:{run/"generations.sqlite3"}?mode=ro',uri=True);origins[arm]=str(run)
    refs={}
    for refarm in ('fix_all','j0'):
        refout,refmarker,refconfig=native(O/'longeval'/refarm/length)
        assert refconfig['driver_options']==options
        refs[refarm]=read(refout/f'longeval_{length}.json')
        assert len(refs[refarm]['records'])==50
    for i in range(50):
        seed=1234+(zlib.crc32(length.encode())%100000)
        prompt,expected,label,nlines=le.build_lines_prompt(le._LENGTH_TOKENS[length],tok,random.Random(seed*1000+i))
        ids=tok.encode(prompt,add_special_tokens=True,return_tensors='pt');bare=tok.encode(f'line {label}',add_special_tokens=False)
        gen={'chunk_size':512,'max_new_tokens':16,'selector':'bm25','topk':12,'sink_tokens':'bos','needle_chunk_set':None,'bare_question_ids':bare,
             'no_retrieval':False,'stats':None,'iter_rounds':0,'iter_hop_topk':2,'iter_score':'meanpool','iter_conf_ratio':.3,'iter_max_chunks':64,'use_kv_cache':True}
        key=key_for(ids,gen)
        for refarm,obj in refs.items():
            rr=obj['records'][i]
            assert rr['sample_index']==i and rr['expected']==expected and rr['label']==label and rr['n_lines']==nlines
        for arm in allarms:
            r=objects[arm]['records'][i]
            assert r['expected']==expected and r['label']==label and r['n_lines']==nlines and r['output']!='[OOM]'
            assert r['pred']==le.extract_prediction(r['output']) and r['correct']==(r['pred']==expected)
            hit=dbs[arm].execute('select prediction,n_tokens from generations where key=?',(key,)).fetchone()
            assert hit and json.loads(hit[0])==r['output'] and hit[1]==ids.numel();le_checks+=1
    scores={}
    for arm in allarms:
        correct=sum(r['correct'] for r in objects[arm]['records']);score=100*correct/50;scores[arm]=score
        near(score/100,objects[arm]['summary']['accuracy']);assert objects[arm]['summary']['correct']==correct and objects[arm]['summary']['total']==50
        dbs[arm].close()
        if arm in targets:d['writeback'].append({'benchmark':'longeval','task':'lines','length':length,'arm':arm,'complete':True,'n':50,'expected_n':50,'correct':correct,
            'metric':'Encbank-adapted first-number exact-match accuracy','score_percent':score,'display':f'{score:.2f}','source':origins[arm],
            'source_token_generation_cache_checks':50,'runtime_selected_pack_persisted':False})
    scores.update({arm:100*obj['summary']['accuracy'] for arm,obj in refs.items()})
    d['comparisons'].append({'benchmark':'longeval','length':length,'scores_percent':scores,'n':50,'new_arm_source_token_cache_verified':True,'reference_source_metadata_equal':True,'reference_rescored_this_round':False,
        'each_prediction_matches_own_cache_entry':True,'runtime_selected_pack_persisted':False,'new_arms':targets})
    print('LongEval verified',length,scores,flush=True)

new_ruler=[];ruler_checks=0
for arm in ('pub','pub_sink','cbos'):
    for p in sorted((O/'ruler'/arm).glob('*.json')):
        obj=read(p);args=obj['args'];task=args['tasks'];length=args['lengths'];jobid=f'ruler__{arm}__{task}_{length}'
        job=state['jobs'].get(jobid,{})
        if job.get('status')!='completed' or job.get('exit_code')!=0:
            d['pending'].append({'benchmark':'ruler','task':task,'length':length,'arm':arm,'complete':False,'n_observed':len(obj['rows']),'score_percent':None,'status':job.get('status')});continue
        if ('ruler',task,length,arm) in known:continue
        assert obj['arms']==[arm] and obj['lora'] is None and args['adapter']==args['adapter_pt']==''
        assert args['model']==str(R/'models/Qwen3-8B') and args['j']==12 and args['n']==50 and args['topk']==12 and args['seed']==42 and args['selector']=='auto' and args['check']==0
        assert args['max_new_tokens']==48 and args['iter_hop_topk']==4
        rr=obj['rows'];assert len(rr)==50 and [r['i'] for r in rr]==list(range(50))
        log=Path(job['log']).read_text(errors='replace');assert 'PYTHONHASHSEED=0' in log and '[remote admission]' in log
        for r in rr:
            assert r['task']==task and r['length']==length and r[arm+'_out']!='[OOM]'
            near(ru._string_match_all_one(r[arm+'_out'],r['answers']),r[arm+'_recall']);ruler_checks+=1
        score=100*sum(r[arm+'_recall'] for r in rr)/50;near(score,obj['summary'][f'{task}/{length}'][arm])
        ref_scores={}
        for refarm in ('fix_all','j0'):
            ref=read(O/'ruler'/refarm/f'{task}_{length}.json')
            assert {k:v for k,v in args.items() if k not in ('arms','out')}=={k:v for k,v in ref['args'].items() if k not in ('arms','out')}
            assert len(ref['rows'])==50
            for a,b in zip(rr,ref['rows']):assert all(a[k]==b[k] for k in ('task','length','i','answers','n_tokens'))
            ref_scores[refarm]=ref['summary'][f'{task}/{length}'][refarm]
        cell={'benchmark':'ruler','task':task,'length':length,'arm':arm,'complete':True,'n':50,'expected_n':50,'score_percent':score,'display':f'{score:.2f}',
            'metric':'Encbank RULER adaptation: answer substring recall','source':str(p),'raw_rescores':50,'metadata_paired_with_v2_j0':True,
            'paired_reference_scores_percent':ref_scores,'exact_input_tokens_persisted':False,'runtime_selected_pack_persisted':False,'generation_cache_available':False}
        d['writeback'].append(cell);new_ruler.append(cell)
        print('RULER verified',arm,task,length,score,flush=True)

# Read only complete shards for coverage; never turn partial task means into final scores.
for arm in ('pub','pub_sink','cbos'):
    for task,n in (('longbook_qa_eng',351),('longbook_choice_eng',229)):
        seen={};shards=[]
        for shard in range(4):
            run=O/'infinitebench'/arm/f'{task}_s{shard}of4'
            if not (run/'COMPLETED.json').exists():continue
            out,m,c=native(run);rr=rows(out/f'{task}_{shard}.jsonl');expected=list(range(shard,n,4))
            assert [r['index'] for r in rr]==expected and m['new_generations']+m['reused_generations']==len(expected)
            for r in rr:assert r['index'] not in seen and r['status']=='ok';seen[r['index']]=r['id']
            shards.append(shard)
        assert len(set(seen.values()))==len(seen)
        # A newly full task would require a separate full source/token/pack/scorer audit.
        d['pending'].append({'benchmark':'infinitebench','task':task,'arm':arm,'completed_shards':shards,'n':len(seen),'expected_n':n,
            'coverage_complete':len(seen)==n,'writeback_eligible':False,'score_percent':None})
    seen={};shards=[];counts=Counter()
    for shard in range(10):
        run=O/'locomo'/arm/f'all_s{shard}of10'
        if not (run/'COMPLETED.json').exists():continue
        m=read(run/'COMPLETED.json');c=read(run/'run_config.json');rr=rows(run/'predictions.jsonl')
        assert m['status']=='completed' and m['n']==len(rr)==len(c['expected'])
        assert [{k:r[k] for k in ('index','id','task')} for r in rr]==c['expected']
        assert [r['index'] for r in rr]==list(range(shard,1986,10))
        for r in rr:
            assert r['id'] not in seen and r['status']=='ok';seen[r['id']]=r['index'];counts[r['category']]+=1
        shards.append(shard)
    expected_counts={1:282,2:321,3:96,4:841,5:446}
    d['pending'].append({'benchmark':'locomo','task':'all','arm':arm,'completed_shards':shards,'n':len(seen),'expected_n':1986,'coverage_complete':len(seen)==1986,
        'categories':[{'category':cat,'n':counts[cat],'expected_n':n,'coverage_complete':counts[cat]==n,'score_percent':None} for cat,n in expected_counts.items()],
        'writeback_eligible':False,'score_percent':None})
d['checks']={'longeval_source_token_exact_input_cache_and_rescore_checks':le_checks,'new_ruler_raw_rescores':ruler_checks,'models_constructed':0,
    'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'OMP_NUM_THREADS':os.environ['OMP_NUM_THREADS'],
    'partial_scores_reported':False,'all_new_full_IB_LoCoMo_groups_audited':not any(c.get('coverage_complete') or any(x['coverage_complete'] for x in c.get('categories',[])) for c in d['pending'] if c['benchmark'] in ('infinitebench','locomo'))}
d['finished_at_utc']=datetime.now(timezone.utc).isoformat()
d['narrative_review']={'body_change_required':False,'current_comparator':'full recompute',
    'new_boundary':'LongEval16k: isolated full KV and V2 both82%, versus full recompute96%; no strict V2 quality superiority to full KV.',
    'appendix_action':'Fill existing LongEval rows and explicitly state the16k tie and14pp gap. Keep RULER4k baseline results in existing appendix grid.',
    'no_new_full_IB_LoCoMo_scores':True}
p=B/'heartbeat_extended_20260909_0329.json';p.write_text(json.dumps(d,indent=2)+'\n')
out=['# 03:29 扩展基线准确率独立 CPU 核查','',
     '相对02:29已核结果，仅检查本轮新增的LongEval/RULER基线完整格及InfiniteBench/LoCoMo覆盖情况。全部评分、源prompt重建、tokenizer与SQLite检查在远端CPU执行，CUDA隐藏，未构造模型；未改论文、同步器、队列或GPU任务。', '',
     '## 新完整格','', '| Benchmark / task | Length | Arm | n | Score (%) |', '|---|---|---|---:|---:|']
for c in d['writeback']:out.append(f"| {c['benchmark']} / {c['task']} | {c['length']} | {c['arm']} | {c['n']} | {c['score_percent']:.2f} |")
out+=['','`pub`=原始Encbank；`pub_sink`=Encbank+BOS；`cbos`=Full KV隔离缓存控制，不是CacheBlend。0.00是50例全部失败的实测结果；缺失项仍为空。','',
      '## LongEval证据和边界','',
      f"新格共 **{le_checks}** 次源label/expected/line count、完整token输入与generation参数缓存键及原评分检查通过。每格连续sample_index0–49，completion receipt new+reused=50，实际reader/effective j/sink和共享参数均正确。每个预测只与它自己的精确输入SQLite缓存记录匹配，不要求不同方法输出相同。",'',
      '统一stock Qwen3-8B BF16、SDPA/repeat_kv policy、j12（cbos有效j36；j0为0）、chunk512、BM25 top12、seed1234、生成cap16。源prompt按原driver用crc32长度seed重新合成，greedy完成式输出；评分是首个至少四位数字与六位目标精确匹配。采用Encbank-adapted LongEval名称，不冒称官方LongChat prompt/scorer。', '',
      '**完整输入token和exact-input generation-cache检查不等于runtime selected-pack验证。** 历史LongEval没有持久化selected indices或完整运行时retrieval pack；本轮没有填造它们。固定token/selector/预算提供配对依据，报告仍保留这条限制。','',
      'LongEval配对参照（V2/j0沿用已核分数，本轮只对齐source metadata与共享参数，不重评分、不重查其缓存）：','']
for c in d['comparisons']:out.append(f"- {c['length']}: {c['scores_percent']}，各50例。")
out+=['','## RULER证据和边界','',
      f"新基线共 **{ruler_checks}** 次raw recall重评分通过；每格completed/exit_code0、i=0–49、50例完整，保存summary与重评分一致。与已核V2/j0同格逐行task/length/i/answers/n_tokens相同，CLI仅arm/out不同，日志有PYTHONHASHSEED=0和远端准入记录。",'',
      '同一remote runtime、stock Qwen3-8B、seed42、top12、chunk512；auto对单/多键为BM25、生成cap48，对five-needle为iter_bm25、hop_topk4、cap60。只检查当前新格，不重跑或重新合成模型输入。', '',
      '**legacy RULER未存原始完整token、运行时selected pack或generation-cache key。** 因此只有sample metadata/CLI/seed配对证据；不能升级为exact-input/runtime pack验证，也不把旧Windows/hash/encoding的样本与新remote样本当作同一输入。指标是Encbank RULER适配路径的answer substring recall，five-needle仍不是组合推理证明。', '',
      '## InfiniteBench和LoCoMo：本轮仍无新完整基线任务或类别','',
      '| Benchmark / arm | Completed shard samples | Expected n | Eligible score |','|---|---:|---:|---|']
for c in d['pending']:
    if c['benchmark'] in ('infinitebench','locomo'):out.append(f"| {c['benchmark']} / {c['task']} / {c['arm']} | {c['n']} | {c['expected_n']} | — |")
out+=['','每个已完成shard核查completion与完整索引，跨shard无重复。JSON另列LoCoMo各类别的已完成样本数和官方总数；当前没有任一完整类别。未发布局部均值，也未重新评分尚未完整的任务。上述数字只计完整shard，不包括活动shard已产生但未完成的行。', '',
      '## 紧凑写回建议','',
      '1. 当前Table16(a)保留现有方法行和六长度列，只补本轮8k/16k新格；其余用dash。V2/j0已有完整两行保持。',
      '2. RULER的新基线只有4k等少量格，放在原RULER附录文字或一个紧凑4k行组：列Original/+BOS/Full KV/V2/full recompute。缺失值dash，不扩大正文表，不将cbos命名为CacheBlend。',
      '3. Table16的InfiniteBench和LoCoMo已核V2/j0不变；新baseline未完整，不能加入局部分数。',
      '4. caption继续写LongEval adaptation、RULER legacy配对限制及未完成dash；实测0保留0。', '',
      '## 正文概括是否需要修改','',
      '当前正文明确以full recompute为参照：V2在LongEval仍落后，这一概括保持准确。本轮新增的关键边界是16k isolated full KV与V2同为82%，而full recompute为96%；因此不能声称V2在所有长度严格优于Full KV。建议仅在扩展任务附录明确此平局和14个百分点缺口，不扩大正文结论，也不从这组准确率推算未测的同任务速度优势。', '',
      '机器结果：`heartbeat_extended_20260909_0329.json`；复算脚本：`audit_extended_0329_remote.py`。文件只新增本轮审计和报告。','']
(B/'heartbeat_extended_20260909_0329.md').write_text('\n'.join(out))
print(json.dumps({'writeback':d['writeback'],'checks':d['checks'],'pending_IB_LoCoMo':[c for c in d['pending'] if c['benchmark'] in ('infinitebench','locomo')]}),flush=True)

