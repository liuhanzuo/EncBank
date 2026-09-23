"""Lightweight native BABILong completion receipts; no torch/model/scoring imports."""
import csv,json,math
from pathlib import Path

def completed(job):
    out=Path(job['output']).resolve();receipt=out/'COMPLETED.json'
    if not receipt.exists():return False
    marker=json.loads(receipt.read_text());assert marker.get('status')=='completed','non-complete receipt'
    if job['benchmark']!='babilong':
        if job.get('layout')=='official_qa':assert marker.get('n')==job['expected_n'],'official QA receipt count'
        return True  # Only used as a frontier observation for other native jobs.
    assert len(job['cells'])==1,'one native dataset cell required'
    cell=job['cells'][0];task=cell['task'];length=cell['length'];n=job['expected_n'];arm=job['arm']
    assert n==cell['expected_n']==100 and cell['indices']==list(range(100)),'native expected indices'
    config=json.loads((out/'run_config.json').read_text());opts=config['driver_options']
    assert config['benchmark']=='babilong' and config['arm']==arm,'native config identity'
    expected={'resume_j':12,'baseline':'none','lora_adapter':'','selector':'bm25','topk':4,'chunk_size':512,'sink_tokens':'bos','tasks':[task],'lengths':[length],'dataset_name':'RMT-team/babilong','limit':100,'max_new_tokens':20,'num_shards':1,'shard_index':0,'device':'cuda:0','dtype':'bfloat16','attn_impl':'sdpa','output_dir':'__ATTEMPT_OUTPUT__'}
    assert all(opts.get(k)==v for k,v in expected.items()),'native protocol options'
    assert config['scoring']=='official helper' and config['pack_protocol']=='whole source/no truncation; explicit complete query; no chat wrapper; identical token selection/order; BOS then EOS sink fallback; first-token EOS suppressed','native pack/scorer'
    info=marker['reader'];assert info['arm']==arm and info['requested_j']==12,'reader requested depth'
    if arm in ('fix_all','fix_none'):want=('CoMemLower',12,True)
    elif arm=='cbos':want=('CoMemLower',36,True)
    elif arm=='j0':want=('CoMem',0,False)
    elif arm in ('pub','pub_sink'):want=('CoMem',12,arm=='pub_sink')
    else:raise AssertionError('unsupported arm')
    assert (info['class'],info['effective_j'],info['write_sink'])==want,'native reader implementation'
    native=Path(marker['output_dir']).resolve();rel=native.relative_to(out/'attempts')
    assert len(rel.parts)==2 and rel.parts[0].isdigit() and rel.parts[1]=='native','native attempt output path'
    forwarded=job['argv'][job['argv'].index('--')+1:]
    assert marker['driver_argv']==forwarded+['--out',str(native)],'native exact forwarded argv'
    assert len(marker['files'])==1,'native receipt files'
    item=marker['files'][0];path=Path(item['file']).resolve()
    assert item['records']==n and path.parent==native and path.name==f'{task}_{length}_official_explicit.csv','native file receipt'
    counters=[marker.get('new_generations'),marker.get('reused_generations')]
    assert all(isinstance(x,int) and x>=0 for x in counters) and sum(counters)==n,'native generation count'
    with path.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==n and [int(r['index']) for r in rows]==cell['indices'],'native CSV count/indices'
    assert [r['id'] for r in rows]==[f'{task}/{length}/{i}' for i in cell['indices']],'native CSV source IDs'
    for row in rows:
        assert row['task']==task and row['length']==length and row['status']=='ok' and row['output']!='[OOM]','native source/status'
        score=float(row['score']);assert math.isfinite(score) and 0<=score<=1,'native score range'
    return True
