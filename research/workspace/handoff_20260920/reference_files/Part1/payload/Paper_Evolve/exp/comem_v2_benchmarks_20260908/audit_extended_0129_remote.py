"""Bounded extended-task audit on remote CPU; never constructs a model."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
import sys, json, sqlite3, hashlib, random, zlib, re
from pathlib import Path
from datetime import datetime, timezone
R = Path('/data/liuhanzuo/comem_v2_20260908')
B = R/'workspace/exp/comem_v2_benchmarks_20260908'
sys.path[:0] = [str(B), str(R/'workspace/COMem'), str(R/'workspace/exp')]
import prepare_infinitebench as ib
from benchmark_pack import BOUNDARY, tokenize_explicit_prompt
from transformers import AutoTokenizer
from eval import longeval as le
from eval import ruler as ruler

def read(p): return json.loads(p.read_text())
def lines(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def near(a,b): assert abs(a-b)<1e-9,(a,b)
def cache_key(tokens, generation):
    return hashlib.sha256(json.dumps({'tokens':tokens,'generation':generation},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def db_for(run): return sqlite3.connect(f'file:{run / "generations.sqlite3"}?mode=ro',uri=True)
def check_db(db,key,pred,ntokens):
    row=db.execute('select prediction,n_tokens from generations where key=?',(key,)).fetchone()
    assert row and json.loads(row[0])==pred and row[1]==ntokens
def native(run):
    marker=read(run/'COMPLETED.json');assert marker['status']=='completed'
    return Path(marker['output_dir']),marker,read(run/'run_config.json')

state=read(R/'outputs/queue_v2/full/state.json')
O=R/'outputs/benchmarks_v2/full'
report={'timestamp':datetime.now(timezone.utc).isoformat(),'execution':'remote CPU only, CUDA_VISIBLE_DEVICES empty, no model construction',
        'scope':'New InfiniteBench choice and RULER multi-key256k, missing Table16 RULER cells, newly completed LongEval pub4k; excludes LongBench/LoCoMo/BABILong',
        'writeback':[],'comparisons':[],'pending':[],'ruler_matrix':[],'checks':{}}
tok=AutoTokenizer.from_pretrained(R/'models/Qwen3-8B',local_files_only=True)
task='longbook_choice_eng';n=229
refs={};dbs={};base_options=None;source_paths={}
for arm in ('fix_all','j0'):
    refs[arm]={};source_paths[arm]=[]
    for shard in range(4):
        run=O/'infinitebench'/arm/f'{task}_s{shard}of4'
        out,marker,config=native(run);rows=lines(out/f'{task}_{shard}.jsonl');opt=config['driver_options']
        indices=list(range(shard,n,4));assert [r['index'] for r in rows]==indices
        assert marker['new_generations']+marker['reused_generations']==len(indices)
        assert config['scoring']=='official helper' and config['arm']==arm
        assert opt['tasks']==[task] and opt['num_shards']==4 and opt['shard_index']==shard and opt['max_samples']==-1
        assert opt['model_path']==str(R/'models/Qwen3-8B') and opt['dtype']=='bfloat16' and opt['attn_impl']=='sdpa'
        assert opt['resume_j']==12 and opt['chunk_size']==512 and opt['topk']==12 and opt['selector']=='bm25'
        assert opt['prompt_style']=='yarn-mistral' and opt['max_new_tokens'] is None and opt['baseline']=='none' and opt['lora_adapter']==''
        shared={k:v for k,v in opt.items() if k!='shard_index'}
        if base_options is None:base_options=shared
        assert shared==base_options
        reader=marker['reader'];assert reader['arm']==arm and reader['effective_j']==(12 if arm=='fix_all' else 0)
        assert reader['class']==('CoMemLower' if arm=='fix_all' else 'CoMem')
        assert reader['kernel_policy']=='s15 repeat_kv: use_gqa_in_sdpa=False (all arms)'
        assert reader['model_config']['model_type']=='qwen3'
        score=read(out/'scores.json')[task];assert score['n']==len(indices)
        near(score['score'],sum(r['score'] for r in rows)/len(rows))
        for row in rows:assert row['index'] not in refs[arm];refs[arm][row['index']]=row
        dbs[arm,shard]=db_for(run);source_paths[arm].append(str(run))
    assert sorted(refs[arm])==list(range(n)) and len({r['id'] for r in refs[arm].values()})==n

stats={'context_tokens':[],'query_tokens':[],'read_pack_tokens':[],'input_tokens':[]}
paired_keys=[];paired_counts={'both_correct':0,'v2_only':0,'j0_only':0,'neither':0}
for i,src in enumerate(ib.iter_task(task)):
    prompt=ib.format_prompt(dict(src,context=src['context']+BOUNDARY),task,prompt_style='yarn-mistral')
    ids,nctx,selected,pack=tokenize_explicit_prompt(tok,prompt,str(src.get('input') or src.get('question') or ''),512,'bm25',12)
    generation={'context_token_count':nctx,'selected_indices':selected,'chunk_size':512,'max_new_tokens':40}
    key=cache_key(ids.tolist(),generation);paired_keys.append(key)
    for arm in ('fix_all','j0'):
        row=refs[arm][i]
        assert row['id']==src['id'] and row['answers']==ib.get_answer(src,task) and row['task']==task and row['status']=='ok'
        assert row['truncation']=='none' and row['prompt_style']=='yarn-mistral' and row['input_tokens']==ids.numel() and row['pack']==pack
        value=float(ib.score_prediction(row['pred'],src,task));near(value,row['score']);assert value in (0.,1.)
        check_db(dbs[arm,i%4],key,row['pred'],ids.numel())
    a,b=refs['fix_all'][i]['score'],refs['j0'][i]['score']
    paired_counts['both_correct' if a and b else 'v2_only' if a else 'j0_only' if b else 'neither']+=1
    for k in stats:stats[k].append(pack[k])
    if (i+1)%50==0:print(f'IB exact source/token/pack/cache {i+1}/{n}',flush=True)
assert len(paired_keys)==n
for db in dbs.values():db.close()
for arm in refs:
    correct=int(sum(r['score'] for r in refs[arm].values()));score=100*correct/n
    report['writeback'].append({'benchmark':'infinitebench','task':task,'arm':arm,'n':n,'expected_n':n,'correct':correct,'complete':True,'metric':'official option accuracy',
        'score_percent':score,'display':f'{score:.2f}','official_rescores':n,'exact_source_token_pack_cache_checks':n,'sources':source_paths[arm]})
report['comparisons'].append({'task':task,'v2_minus_j0_pp':100*sum(refs['fix_all'][i]['score']-refs['j0'][i]['score'] for i in range(n))/n,
    'paired_counts':paired_counts,'same_exact_input_generation_key_all_examples':True,'generation_cap':40,'significance_test_performed':False,
    'token_ranges':{k:{'min':min(v),'max':max(v)} for k,v in stats.items()},'keys_sha256':hashlib.sha256('\n'.join(paired_keys).encode()).hexdigest()})
print('IB full comparison passed',flush=True)

# Light legacy rescore. RULER does not have persisted token sequences, retrieval packs,
# or GenerationCache databases: do not upgrade metadata pairing into exact-pack proof.
cells=[('niah_single_2','4k'),('niah_multikey_1','4k'),('variable_tracking','4k'),('variable_tracking','8k'),
       ('variable_tracking','64k'),('niah_single_2','128k'),('niah_multikey_1','128k'),('variable_tracking','128k'),
       ('niah_single_2','256k'),('niah_multikey_1','256k')]
for task,length in cells:
    paired={};opts=None;scores={};source=[]
    for arm in ('fix_all','j0'):
        job=state['jobs'][f'ruler__{arm}__{task}_{length}'];assert job['status']=='completed' and job['exit_code']==0
        p=O/'ruler'/arm/f'{task}_{length}.json';obj=read(p);rows=obj['rows'];args=obj['args']
        assert len(rows)==50 and [r['i'] for r in rows]==list(range(50)) and obj['arms']==[arm] and obj['lora'] is None
        assert args['model']==str(R/'models/Qwen3-8B') and args['j']==12 and args['n']==50 and args['topk']==12 and args['seed']==42 and args['check']==0
        assert args['selector']=='auto' and args['max_new_tokens']==48 and args['adapter']==args['adapter_pt']==''
        comparable={k:v for k,v in args.items() if k not in ('arms','out')}
        if opts is None:opts=comparable
        assert opts==comparable
        log=Path(job['log']).read_text(errors='replace');assert 'PYTHONHASHSEED=0' in log and '[remote admission]' in log
        for row in rows:
            assert row['task']==task and row['length']==length and row[arm+'_out']!='[OOM]'
            value=ruler._string_match_all_one(row[arm+'_out'],row['answers']);near(value,row[arm+'_recall'])
        score=100*sum(r[arm+'_recall'] for r in rows)/50;near(score,obj['summary'][f'{task}/{length}'][arm])
        paired[arm]=rows;scores[arm]=score;source.append(str(p))
    for a,b in zip(paired['fix_all'],paired['j0']):assert all(a[k]==b[k] for k in ('task','length','i','answers','n_tokens'))
    cell={'benchmark':'ruler','task':task,'length':length,'n':50,'metric':'answer substring recall (CoMem RULER adaptation)',
        'scores_percent':scores,'delta_v2_minus_j0_pp':scores['fix_all']-scores['j0'],'complete':True,'rescores':100,
        'paired_sample_metadata':True,'same_args_except_arm_out':True,'logged_pythonhashseed':0,
        'exact_original_tokens_verified':False,'exact_original_selected_pack_verified':False,'generation_cache_key_available':False,
        'n_tokens_range':[min(r['n_tokens'] for r in paired['fix_all']),max(r['n_tokens'] for r in paired['fix_all'])],
        'new_complete_pair_since_0027':task=='niah_multikey_1' and length=='256k','sources':source}
    report['ruler_matrix'].append(cell)
    if cell['new_complete_pair_since_0027']:
        for arm,value in scores.items():report['writeback'].append({'benchmark':'ruler','task':task,'length':length,'arm':arm,'n':50,'expected_n':50,
            'complete':True,'score_percent':value,'display':f'{value:.2f}','metric':cell['metric'],'exact_pack_claim_allowed':False,'sources':source})
print('RULER missing-table matrix rescored',flush=True)

# New pub 4k LongEval cell, paired to existing V2/j0 with exact reconstructed
# token inputs and the per-method successful-generation cache for all 50 samples.
run=O/'longeval/pub/4k'
if (run/'COMPLETED.json').exists():
    old={};dbs={};common=None;sources=[]
    for arm in ('pub','fix_all','j0'):
        run=O/'longeval'/arm/'4k';out,marker,config=native(run);opt=config['driver_options'];obj=read(out/'longeval_4k.json')
        assert marker['new_generations']+marker['reused_generations']==50 and len(obj['records'])==50
        assert opt['seed']==1234 and opt['max_new_tokens']==16 and opt['topk']==12 and opt['chunk_size']==512 and opt['selector']=='bm25'
        assert opt['lengths']==['4k'] and opt['num_samples']==50 and opt['baseline']=='none' and opt['lora_adapter']==''
        assert config['scoring']=='unchanged legacy driver; do not relabel as official'
        if common is None:common=opt
        assert opt==common
        old[arm]=obj;dbs[arm]=db_for(run);sources.append(str(run))
    for i in range(50):
        seed=1234+(zlib.crc32(b'4k')%100000)
        prompt,expected,label,nlines=le.build_lines_prompt(4096,tok,random.Random(seed*1000+i))
        ids=tok.encode(prompt,add_special_tokens=True,return_tensors='pt');bare=tok.encode(f'line {label}',add_special_tokens=False)
        generation={'chunk_size':512,'max_new_tokens':16,'selector':'bm25','topk':12,'sink_tokens':'bos','needle_chunk_set':None,
                    'bare_question_ids':bare,'no_retrieval':False,'stats':None,'iter_rounds':0,'iter_hop_topk':2,'iter_score':'meanpool',
                    'iter_conf_ratio':.3,'iter_max_chunks':64,'use_kv_cache':True}
        key=cache_key(ids.tolist(),generation)
        for arm in old:
            row=old[arm]['records'][i];assert row['sample_index']==i and row['expected']==expected and row['label']==label and row['n_lines']==nlines and row['output']!='[OOM]'
            assert row['pred']==le.extract_prediction(row['output']) and row['correct']==(row['pred']==expected)
            check_db(dbs[arm],key,row['output'],ids.numel())
    scores={arm:100*sum(r['correct'] for r in obj['records'])/50 for arm,obj in old.items()}
    for arm,obj in old.items():near(scores[arm]/100,obj['summary']['accuracy']);dbs[arm].close()
    report['writeback'].append({'benchmark':'longeval','task':'lines','length':'4k','arm':'pub','n':50,'expected_n':50,'complete':True,'metric':'CoMem-adapted lines accuracy',
        'score_percent':scores['pub'],'display':f"{scores['pub']:.2f}",'sources':sources,'source_token_cache_checks':150,'recorded_pack_available':False})
    report['comparisons'].append({'task':'longeval lines/4k','n':50,'scores_percent':scores,'same_source_token_inputs_and_generation_keys':True,
        'historical_selected_indices_recorded':False,'official_LongChat_protocol':False})
    print('LongEval pub4k exact token/cache checks passed',flush=True)

# Bounded completion inventory from real remote files; partial benchmark scores are null.
for arm in ('pub','pub_sink','cbos'):
    for task,n in (('longbook_qa_eng',351),('longbook_choice_eng',229)):
        count=0;shards=[]
        for shard in range(4):
            run=O/'infinitebench'/arm/f'{task}_s{shard}of4'
            if (run/'COMPLETED.json').exists():
                out,marker,config=native(run);count+=len(lines(out/f'{task}_{shard}.jsonl'));shards.append(shard)
        report['pending'].append({'benchmark':'infinitebench','task':task,'arm':arm,'n_completed_shards':count,'expected_n':n,
            'completed_shards':shards,'complete':count==n,'score_percent':None,'reason':'No newly full comparison; incomplete tasks stay blank'})
report['checks']={'infinitebench_official_rescores':458,'infinitebench_source_token_pack_cache_checks':458,'ruler_light_rescores':100*len(cells),
    'source_tokenization_on_remote_CPU':True,'models_constructed':0,'CUDA_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
    'ruler_limit':'Raw legacy output stores sample index/answers/token count, not complete token IDs, selected indices, or cache key. Paired metadata + equal CLI + fixed hash seed only; no exact historical pack proof.'}
report['table16_suggestion']={'infinitebench':'Keep En.QA F1 n351, add En.MC official option accuracy n229 (56.77,52.84). Caption says two English long-book tasks, not full InfiniteBench.',
    'ruler':'Replace three scattered RULER rows with 6-row task-by-method matrix, columns 4k/8k/64k/128k/256k; use existing audited 8/64 NIAH cells and this report missing cells. Five-needle256k unplanned/unknown stays dash. If width must stay minimal, columns4k/128k/256k plus footnote five-needle8k92.0 and64k97.6 versus100.',
    'longeval':'Optional new pub row4k only: remaining lengths dash. Keep all V2 shortfalls visible; adapted protocol.'}
dest=B/'heartbeat_extended_20260909_0129.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'writeback':report['writeback'],'comparisons':report['comparisons'],'pending':report['pending'],'output':str(dest)}),flush=True)
