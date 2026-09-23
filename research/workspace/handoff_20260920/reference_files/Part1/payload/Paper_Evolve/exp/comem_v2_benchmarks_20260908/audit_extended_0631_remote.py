"""New extended baseline accuracy cells only; remote CPU, no models."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import sys,json,sqlite3,hashlib,random,zlib
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
R=Path('/data/liuhanzuo/comem_v2_20260908');B=R/'workspace/exp/comem_v2_benchmarks_20260908';O=R/'outputs/benchmarks_v2/full'
sys.path[:0]=[str(B),str(R/'workspace/COMem'),str(R/'workspace/exp')]
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(16)
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
def source_from_recorded_line_count(target,tok,rng,nlines):
    """Rebuild native RNG, verify last stopping boundary, then exact cache keys.

    Only the source reconstruction avoids repeated prefix tokenization. It does
    not change a benchmark, generate a model output, or infer a runtime pack.
    """
    assert nlines>=64 and nlines%64==0
    labels=[];values=[];text_lines=[]
    for _ in range(nlines):
        label=le._random_label(rng);value=str(rng.randint(100000,999999))
        labels.append(label);values.append(value)
        text_lines.append(f'line {label}: REGISTER_CONTENT is <{value}>\n')
    def render(query_label,count):
        query=(f'\nNow the record is over. Tell me what is the <REGISTER_CONTENT> in '
               f'line {query_label}? I need the number.\nThe <REGISTER_CONTENT> in line '
               f'{query_label} is')
        return le._PROMPT_HEADER+''.join(text_lines[:count])+query
    final_probe=len(tok.encode(render(labels[nlines//2],nlines),add_special_tokens=True))
    assert final_probe>=target
    previous_probe=None
    if nlines>64:
        previous_probe=len(tok.encode(render(labels[(nlines-64)//2],nlines-64),add_special_tokens=True))
        assert previous_probe<target
    ti=rng.randrange(nlines)
    return (render(labels[ti],nlines),values[ti],labels[ti],nlines),{'previous_batch_tokens':previous_probe,'stop_batch_tokens':final_probe}
prior={'writeback':read(B/'heartbeat_extended_20260909_0129.json')['writeback']+read(B/'heartbeat_extended_20260909_0229.json')['writeback']+read(B/'heartbeat_extended_20260909_0329.json')['writeback']+read(B/'heartbeat_extended_20260909_0429.json')['writeback']+read(B/'heartbeat_extended_20260909_0531.json')['writeback']}
known={(c['benchmark'],c['task'],c.get('length'),c['arm']) for c in prior['writeback']}
snapshot=read(B/'heartbeat_extended_20260909_0631_snapshot.json')
state=snapshot['state']
eligible_native=set(snapshot['eligible_native_markers'])
eligible_ruler=set(snapshot['eligible_ruler_files'])
d={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'execution':'remote CPU; CUDA hidden; no model construction or GPU tasks',
   'writeback':[],'comparisons':[],'pending':[],'checks':{},'prior_reference':['heartbeat_extended_20260909_0129.json','heartbeat_extended_20260909_0229.json','heartbeat_extended_20260909_0329.json','heartbeat_extended_20260909_0429.json','heartbeat_extended_20260909_0531.json']}
d['snapshot_timestamp_utc']=snapshot['timestamp_utc']
d['snapshot_file']=str(B/'heartbeat_extended_20260909_0631_snapshot.json')
tok=AutoTokenizer.from_pretrained(R/'models/Qwen3-8B',local_files_only=True)
new_le=[]
for arm in ('pub','pub_sink','cbos'):
    for length in ('4k','8k','16k','32k','64k','128k'):
        run=O/'longeval'/arm/length
        if str(run/'COMPLETED.json') not in eligible_native:
            d['pending'].append({'benchmark':'longeval','task':'lines','length':length,'arm':arm,'complete':False,'score_percent':None});continue
        if ('longeval','lines',length,arm) not in known:new_le.append((arm,length))
le_checks=0;reconstruction_checks=[]
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
        assert ri['class']==('CoMemLower' if arm in ('fix_all','cbos') else 'CoMem')
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
        sample,boundary=source_from_recorded_line_count(le._LENGTH_TOKENS[length],tok,random.Random(seed*1000+i),objects[targets[0]]['records'][i]['n_lines'])
        prompt,expected,label,nlines=sample
        if i==0:
            assert le.build_lines_prompt(le._LENGTH_TOKENS[length],tok,random.Random(seed*1000+i))==sample
        reconstruction_checks.append({'length':length,'sample_index':i,'recorded_n_lines':nlines,'native_builder_equivalence':i==0,**boundary})
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
        if (i+1)%10==0:print('LongEval source/token/cache progress',length,i+1,50,flush=True)
    scores={}
    for arm in allarms:
        correct=sum(r['correct'] for r in objects[arm]['records']);score=100*correct/50;scores[arm]=score
        near(score/100,objects[arm]['summary']['accuracy']);assert objects[arm]['summary']['correct']==correct and objects[arm]['summary']['total']==50
        dbs[arm].close()
        if arm in targets:d['writeback'].append({'benchmark':'longeval','task':'lines','length':length,'arm':arm,'complete':True,'n':50,'expected_n':50,'correct':correct,
            'metric':'CoMem-adapted first-number exact-match accuracy','score_percent':score,'display':f'{score:.2f}','source':origins[arm],
            'source_token_generation_cache_checks':50,'runtime_selected_pack_persisted':False})
    scores.update({arm:100*obj['summary']['accuracy'] for arm,obj in refs.items()})
    d['comparisons'].append({'benchmark':'longeval','length':length,'scores_percent':scores,'n':50,'new_arm_source_token_cache_verified':True,'reference_source_metadata_equal':True,'reference_rescored_this_round':False,
        'each_prediction_matches_own_cache_entry':True,'runtime_selected_pack_persisted':False,'new_arms':targets})
    print('LongEval verified',length,scores,flush=True)

new_ruler=[];ruler_checks=0
for arm in ('pub','pub_sink','cbos'):
    for p in sorted((O/'ruler'/arm).glob('*.json')):
        if str(p) not in eligible_ruler:continue
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
            'metric':'CoMem RULER adaptation: answer substring recall','source':str(p),'raw_rescores':50,'metadata_paired_with_v2_j0':True,
            'paired_reference_scores_percent':ref_scores,'exact_input_tokens_persisted':False,'runtime_selected_pack_persisted':False,'generation_cache_available':False}
        d['writeback'].append(cell);new_ruler.append(cell)
        print('RULER verified',arm,task,length,score,flush=True)

# Read only complete shards for coverage; never turn partial task means into final scores.
for arm in ('pub','pub_sink','cbos'):
    for task,n in (('longbook_qa_eng',351),('longbook_choice_eng',229)):
        if ('infinitebench',task,None,arm) in known:continue
        seen={};shards=[]
        for shard in range(4):
            run=O/'infinitebench'/arm/f'{task}_s{shard}of4'
            if str(run/'COMPLETED.json') not in eligible_native:continue
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
        if str(run/'COMPLETED.json') not in eligible_native:continue
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
d['longeval_source_reconstruction']={'method':'Native RNG reconstructed using recorded n_lines; final and previous64-line stopping boundaries validated; one source per length compared exactly to native builder; every new output checked by exact-token own-cache key',
    'samples':reconstruction_checks,'old_outputs_rescored':False}
d['finished_at_utc']=datetime.now(timezone.utc).isoformat()
d['narrative_review']={'body_change_required':False,'current_comparator':'full recompute',
    'retained_boundary':'Previously verified LongEval16k: isolated full KV and V2 both82%, versus full recompute96%; no strict V2 quality superiority to full KV.',
    'appendix_action':'Fill existing LongEval rows and explicitly state the16k tie and14pp gap. Keep RULER4k baseline results in existing appendix grid.',
    'no_new_full_IB_LoCoMo_scores':True}
d['checks'].update(torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),cuda_initialized=torch.cuda.is_initialized(),snapshot_fixed=True)
assert d['checks']['all_new_full_IB_LoCoMo_groups_audited'], 'New full task/category requires separate exact source/token/pack audit before completion'
assert not torch.cuda.is_initialized()
d['narrative_review']={
    'body_change_required':False,
    'retained_boundary':'LongEval V2 trails replay at all six lengths; isolated KV ties V2 at16k (82 vs82, replay96). Exact input-token/cache checks do not reconstruct missing runtime packs.',
    'no_new_full_IB_LoCoMo_scores':True,
    'appendix_action':'Fill only the four newly verified128k LongEval/RULER baseline cells. Keep existing negative results and all missing task/category means blank.'}
p=B/'heartbeat_extended_20260909_0631.json';assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
md=['# 06:31 扩展基线准确率独立 CPU 核查','',
    f"固定一次实际快照：{snapshot['timestamp_utc']} UTC（北京时间06:34:00）。仅纳入该快照中主队列completed/exit_code0且具有完整输出的单元；不追收此后完成的任务。旧已核评分不重复做。",'',
    '所有源重建、分词、缓存键和重评分均在远端CPU完成，CUDA隐藏、Torch2线程/interop16、OMP2/MKL2；没有加载模型或启动GPU，没有改论文、队列或同步器。','',
    '## 新增完整格','',
    '| Benchmark / task | Length | Arm | n | Score (%) |','|---|---|---|---:|---:|']
for c in d['writeback']:md.append(f"| {c['benchmark']} / {c['task']} | {c['length']} | {c['arm']} | {c['n']} | {c['score_percent']:.2f} |")
md+=['',f"合计{len(d['writeback'])}个新格、{sum(c['n'] for c in d['writeback'])}个新预测。pub=原始CoMem，pub_sink=CoMem+BOS，cbos=isolated full KV控制，不是CacheBlend。实测0保留0，未知分数保持空白。",'',
    '## LongEval：逐例源/token/自身缓存验证，保留缺失runtime pack边界','',
    f"{le_checks}个新输出通过完整源label/target/n_lines、原始first-number scorer、完整输入token及自身SQLite精确generation-cache键核验。每格50例连续sample_index0–49，completion new+reused=50，共享参数及实际reader/split/sink/kernel policy一致。",'',
    '统一stock Qwen3-8B BF16，SDPA且repeat_kv/禁GQA，j12（cbos有效j36），chunk512、BM25 top12、seed1234、cap16。这里是CoMem-adapted LongEval的completion prompt与首数字精确匹配，不能称为官方LongChat prompt/scorer。', '',
    f"沿用已批准的CPU源重建：按保存n_lines重建native RNG的所有label/value和最终query，逐源检查前一64行批次未达token目标且最终批次达到目标。{len(reconstruction_checks)}个独立源均检查停止边界；{sum(r['native_builder_equivalence'] for r in reconstruction_checks)}个代表例直接与native builder逐字等价。所有新预测仍逐例验证完整tokens和各自缓存键，不以代表例替代逐例验证。",'',
    '历史LongEval未持久化运行时selected indices或retrieval pack。完整输入token与自身generation-cache检查不能补出这些记录，也不能升级为runtime-pack核验。V2/j0只沿用旧已核分数并对齐源metadata和共享参数，本轮未重新评分旧输出。','']
for c in d['comparisons']:md.append(f"- {c['length']}，各n50：{c['scores_percent']}。")
md+=['','## RULER：完整raw重评分，保留legacy限制','',
    f"{ruler_checks}个新输出按原CoMem RULER answer-substring recall重评分，50例/格与summary一致；快照job completed/exit_code0、连续i=0–49、非OOM、CLI/reader参数和准入日志均通过。与已核V2/j0逐例task/length/i/answers/n_tokens一致；旧参考输出不重新评分。",'',
    '原RULER日志未保存完整输入token、运行时selected pack或generation-cache键；配对仍仅限同remote runtime下sample metadata/CLI/seed。不能升级为exact-token/pack配对，也不把旧Windows与新remote的独立draw合并。单针/多键BM25 cap48；five-needle iter_BM25/hop4 cap60，仍是检索而非事实组合。','']
for c in d['writeback']:
    if c['benchmark']=='ruler':md.append(f"- {c['task']}/{c['length']}/{c['arm']}={c['score_percent']:.2f}；旧已核参考{c['paired_reference_scores_percent']}，各n50。")
md+=['','## 未完整任务与类别：仅报告覆盖','',
    '| Benchmark / task / arm | Complete-shard n | Formal n | Eligible score |','|---|---:|---:|---|']
for c in d['pending']:
    if c['benchmark'] in ('infinitebench','locomo'):md.append(f"| {c['benchmark']} / {c['task']} / {c['arm']} | {c['n']} | {c['expected_n']} | — |")
md+=['','覆盖只计固定快照中完整shard，逐shard核receipt/完整索引、跨shard去重及类别分母；活动shard不计。机器JSON列LoCoMo各类别的已完成n和完整类别n。此快照无新增完整InfiniteBench MC任务或LoCoMo类别，不计算局部均值；已核InfiniteBench QA三基线各351例不重复评分。', '',
    '## 写回与不利结果','',
    '只补现有附录LongEval和RULER128k基线格，不新增正文表。LongEval全部三基线六长度现已齐；V2仍低于replay，16k与isolated KV的82/82平局继续保留。新RULER单针的BOS和isolated KV均为100%，与V2/replay并列，不能改写为V2全面领先。所有差值为描述性均值，不作显著性或跨模型架构因果推断；准确率不能外推未测同任务系统速度。','',
    '机器结果：`heartbeat_extended_20260909_0631.json`。固定快照：`heartbeat_extended_20260909_0631_snapshot.json`。复算脚本：`audit_extended_0631_remote.py`。','']
(B/'heartbeat_extended_20260909_0631.md').write_text('\n'.join(md))
print(json.dumps({'writeback':d['writeback'],'checks':d['checks'],'pending_IB_LoCoMo':[c for c in d['pending'] if c['benchmark'] in ('infinitebench','locomo')]}),flush=True)

