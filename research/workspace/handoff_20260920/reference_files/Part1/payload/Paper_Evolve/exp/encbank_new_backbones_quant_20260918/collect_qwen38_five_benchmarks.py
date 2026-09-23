"""Collect verified full 27B quantization outputs and independently aggregate five metrics."""
import collections,datetime,json,math,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918'
MODEL='Qwen3.8-27B';TAG='quant-complete-20260919'
ORDER=('ruler','longeval','longbench','babilong','locomo')
EXPECTED=dict(zip(ORDER,(1500,500,1150,2100,1986)))
def main():
    folders=['results/quant-cell/'+MODEL+'/cell']+['results/quant-full/'+MODEL+'/shard%d'%i for i in range(4)]
    tasks=['quant-cell-m1']+['quant-full-m1-s%d'%i for i in range(4)]
    members=[f+'/'+n for f in folders for n in ['predictions.jsonl','complete.json','protocol.json']]
    members+=['runs/'+t+'/'+n for t in tasks for n in ['submission.json','parent_exit.json']]
    members+=['delivery/cpu_summary_runs/'+TAG+'/'+n for n in ['summary.json','parent_exit.json']]
    members+=['judge_gpt6_astra/judge_decisions.jsonl','judge_gpt6_astra/protocol.json']
    dest=ROOT/'delivery/qwen38_quant_five_benchmarks';dest.mkdir(parents=True,exist_ok=True)
    archive=dest/'saved_outputs.tar.gz'
    with archive.open('wb') as stdout:
        proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
            'timeout -k 5s 90s tar -czf - -C '+REMOTE+' '+' '.join(members)],stdout=stdout,stderr=subprocess.PIPE,timeout=110)
    assert proc.returncode==0,proc.stderr.decode('utf-8','replace')
    raw=[];receipts=[]
    with tarfile.open(archive,'r:gz') as tar:
        assert set(tar.getnames())==set(members)
        def read(name):return json.load(tar.extractfile(name))
        cpu=read('delivery/cpu_summary_runs/'+TAG+'/parent_exit.json');assert cpu['actual_wait'] and cpu['returncode']==0
        official=read('delivery/cpu_summary_runs/'+TAG+'/summary.json')
        protocol=read('judge_gpt6_astra/protocol.json')
        assert protocol['model']=='gpt-6-astra' and protocol['protocol_id']=='midcache-locomo-gpt6-astra-v1'
        for folder,task in zip(folders,tasks):
            complete=read(folder+'/complete.json');rec=read('runs/'+task+'/submission.json');parent=read('runs/'+task+'/parent_exit.json')
            assert complete['generation_complete'] and complete['failures']=={'ok':complete['records']}
            assert parent['actual_wait'] and parent['returncode']==0 and parent['completion_exists']
            p=read(folder+'/protocol.json');assert p['rank']==32 and p['alpha']==32 and p['step']==4000
            rows=[json.loads(line) for line in tar.extractfile(folder+'/predictions.jsonl')];assert len(rows)==complete['records']
            raw.extend(r for r in rows if r['arm'] in ('cache_h8','cache_h4'))
            receipts.append(dict(task=task,job=rec['job'],parent_exit=parent,complete=complete))
        votes=[json.loads(l) for l in tar.extractfile('judge_gpt6_astra/judge_decisions.jsonl')]
        votes=[v for v in votes if v['cohort']==MODEL and v['arm'] in ('cache_h8','cache_h4')]
    assert len(raw)==14472 and len({(r['arm'],r['id']) for r in raw})==14472
    assert len(votes)==3972
    judge={(r['arm'],r['id']):r for r in votes};assert len(judge)==3972
    acct=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
        'sacct -j '+','.join(r['job'] for r in receipts)+' -n -P -o JobIDRaw,State,ExitCode,End'],capture_output=True,text=True,timeout=40)
    assert acct.returncode==0,acct.stderr
    accounting={l.split('|')[0]:l.split('|')[1:3] for l in acct.stdout.splitlines()}
    for rec in receipts:assert accounting[rec['job']]==['COMPLETED','0:0'],rec
    result={}
    for arm in ['cache_h16','cache_h8','cache_h4']:
        data=official['table'][MODEL][arm];assert set(data)==set(ORDER)
        for b,n in EXPECTED.items():assert data[b]['complete'] and data[b]['generated']==data[b]['scored']==n
        result[arm]={b:data[b]['score'] for b in ORDER};result[arm]['avg']=sum(result[arm][b] for b in ORDER)/5
    breakdown={}
    for arm in ['cache_h8','cache_h4']:
        for b,n in EXPECTED.items():
            rows=[r for r in raw if r['arm']==arm and r['benchmark']==b]
            assert len(rows)==n and all(r['status']=='ok' for r in rows)
            cells=collections.defaultdict(list)
            for row in rows:
                score=row['score']
                if b=='locomo':
                    v=judge[(arm,row['id'])];assert v['model']=='gpt-6-astra' and v['protocol_id']==protocol['protocol_id']
                    assert v['judge_correct'] in (0,1);score=v['judge_correct']
                assert isinstance(score,(float,int)) and math.isfinite(score)
                cells[(row['task'],row['length'])].append(score)
            if b=='locomo':value=100*sum(sum(v) for v in cells.values())/n
            else:
                assert len(cells)==dict(ruler=15,longeval=5,longbench=6,babilong=21)[b]
                value=100*sum(sum(v)/len(v) for v in cells.values())/len(cells)
            assert math.isclose(value,result[arm][b],abs_tol=1e-10)
        vs=[v for v in votes if v['arm']==arm]
        breakdown[arm]={}
        for label,categories,n in [('C1-4',{1,2,3,4},1540),('C5',{5},446),('all',{1,2,3,4,5},1986)]:
            part=[v for v in vs if int(v['category']) in categories];assert len(part)==n
            breakdown[arm][label]=dict(n=n,score=100*sum(v['judge_correct'] for v in part)/n)
    report=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),model=MODEL,j=21,
        adapter='rank32 alpha32 final4000',expected_per_arm=EXPECTED,scores=result,locomo_breakdown=breakdown,
        predictions=14472,judge_records=3972,errors=0,remote_cpu_rescored=official['cpu_rescored'],
        cpu_summary_parent_exit=cpu,collection_actual_returncode=proc.returncode,local_aggregation_verified=True,
        source_archive=str(archive),scope='Full five-benchmark H-cache quantization; new rank128 training reported separately.')
    (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (dest/'exit_evidence.json').write_text(json.dumps(dict(receipts=receipts,sacct=acct.stdout,actual_accounting_returncode=acct.returncode),indent=2),encoding='utf-8')
    (dest/'judge_decisions.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in votes),encoding='utf-8')
    print(json.dumps(report))
if __name__=='__main__':main()
