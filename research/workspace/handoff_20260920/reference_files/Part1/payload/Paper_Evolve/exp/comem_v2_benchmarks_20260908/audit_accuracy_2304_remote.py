"""CPU-only audit of the newly complete BABILong qa1/2k pair; no model load."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import csv
import hashlib
import inspect
import json
import sqlite3
import sys
from datetime import datetime, timezone

ROOT = Path('/data/liuhanzuo/comem_v2_20260908')
B = ROOT/'workspace/exp/comem_v2_benchmarks_20260908'
sys.path[:0] = [str(B), str(ROOT/'workspace/COMem')]
from prepare_babilong import load_task, format_prompt, score_prediction
from benchmark_pack import BOUNDARY, tokenize_explicit_prompt
from reader_adapter import ExplicitPackReader, _plain
from transformers import AutoTokenizer

def read(p): return json.loads(p.read_text())

source = load_task('qa1', '2k')
assert len(source) == 100
tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-8B', local_files_only=True)
method = ExplicitPackReader.__new__(ExplicitPackReader).generate_from_ids
arm_rows, scores, records = {}, {}, []
configs = {}
for arm in ('fix_all', 'j0'):
    run = ROOT/'outputs/benchmarks_v2/full/babilong'/arm/'qa1_2k'
    receipt, config = read(run/'COMPLETED.json'), read(run/'run_config.json')
    assert receipt['status'] == 'completed'
    assert receipt['new_generations'] == 100 and receipt['reused_generations'] == 0
    assert receipt['reader']['class'] == ('CoMemLower' if arm == 'fix_all' else 'CoMem')
    assert receipt['reader']['requested_j'] == 12
    assert receipt['reader']['effective_j'] == (12 if arm == 'fix_all' else 0)
    opts = config['driver_options']; configs[arm] = opts
    assert opts['selector']=='bm25' and opts['topk']==4 and opts['chunk_size']==512
    assert opts['tasks']==['qa1'] and opts['lengths']==['2k'] and opts['max_new_tokens']==20
    assert opts['limit']==100 and opts['num_shards']==1 and opts['shard_index']==0
    assert opts['dtype']=='bfloat16' and not opts['lora_adapter'] and opts['baseline']=='none'
    assert len(receipt['files'])==1 and receipt['files'][0]['records']==100
    output=Path(receipt['files'][0]['file'])
    with output.open(newline='') as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==100 and [int(row['index']) for row in rows]==list(range(100))
    assert [row['id'] for row in rows]==[f'qa1/2k/{i}' for i in range(100)]
    db=sqlite3.connect(f'file:{run / "generations.sqlite3"}?mode=ro',uri=True)
    assert db.execute('select count(*) from generations').fetchone()[0]==100
    values=[]
    for i,(row,sample) in enumerate(zip(rows,source)):
        assert row['target']==sample['target'] and row['question']==sample['question'] and row['status']=='ok'
        assert row['task']=='qa1' and row['length']=='2k' and row['output']!='[OOM]'
        value=score_prediction(row['output'],sample,'qa1')
        assert value in (0.,1.) and value==float(row['score'])
        values.append(value)
        marked=format_prompt(dict(sample,input=sample['input'].strip()+BOUNDARY),'qa1')
        ids,count,selected,pack=tokenize_explicit_prompt(tokenizer,marked,sample['question'],512,'bm25',4,None,0,2,'meanpool')
        assert json.loads(row['pack'])==pack
        bound=inspect.signature(method).bind(ids,context_token_count=count,selected_indices=selected,chunk_size=512,max_new_tokens=20)
        bound.apply_defaults(); generation=dict(bound.arguments); generation.pop('input_ids')
        content={'tokens':_plain(ids),'generation':_plain(generation)}
        key=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        cached=db.execute('select prediction,n_tokens from generations where key=?',(key,)).fetchone()
        assert cached is not None and json.loads(cached[0])==row['output'] and cached[1]==ids.numel()
    db.close()
    score=100*sum(values)/100; scores[arm]=score; arm_rows[arm]=rows
    records.append({'benchmark':'babilong','task':'qa1','length':'2k','arm':arm,'n':100,'expected_n':100,
        'score_percent':score,'display':f'{score:.2f}','completed':True,'official_rescore_matches':100,
        'source_id_checks':100,'reconstructed_pack_checks':100,'exact_generation_cache_checks':100,
        'source':str(output),'completion_receipt':str(run/'COMPLETED.json'),
        'finished_at':receipt['finished_at'],'generation_cap':20})
assert configs['fix_all']==configs['j0']
for left,right in zip(arm_rows['fix_all'],arm_rows['j0']):
    for key in ('index','id','task','length','target','question','pack'): assert left[key]==right[key]
result={'timestamp':datetime.now(timezone.utc).isoformat(),'execution':'remote CPU only; CUDA hidden; tokenizer only; no model',
    'writeback':records,'comparisons':[{'benchmark':'babilong','task':'qa1','length':'2k','n':100,
    'v2_minus_j0_pp':scores['fix_all']-scores['j0'],'paired_source_and_pack':True}],
    'protocol':{'data':str(B/'data/babilong/qa1/2k.json'),'prompt':'vendored official DEFAULT_PROMPTS qa1, DEFAULT_TEMPLATE; instruction/examples/post-prompt',
    'chat_template':False,'thinking':'no chat thinking switch used; bare official prompt',
    'truncation':'none; context/query separately tokenized at explicit boundary; no padding',
    'selector':'BM25 top4, chunk512; same bare question and ordered selected chunks',
    'generation':'greedy native explicit reader; max_new_tokens20; stock Qwen3-8B BF16; j12 V2 vs j0',
    'metric':str(B/'vendor/babilong/babilong/metrics.py')+'::compare_answers / TASK_LABELS[qa1]'},
    'limitations':['qa1 requires one supporting fact; this is not qa2/qa3 multi-fact evidence.',
    'The completed 2k cell is not completion of the full BABILong task-by-length grid.',
    'No latency fields are read or reported from remote generation caches.']}
dest=ROOT/'outputs/diagnostics/heartbeat_accuracy_20260908_2304_babilong.json'
dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
