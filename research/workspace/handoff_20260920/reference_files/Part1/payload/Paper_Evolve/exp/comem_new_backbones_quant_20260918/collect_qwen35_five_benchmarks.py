"""Collect completed 9B outputs and verify all five complete benchmark aggregates."""
import collections
import datetime
import json
import math
import subprocess
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918'
ORDER=('ruler','longeval','longbench','babilong','locomo')
EXPECTED=dict(zip(ORDER,(1500,500,1150,2100,1986)))


def main():
    folders=['results/quant-cell/Qwen3.5-9B/cell']+[
        'results/quant-full/Qwen3.5-9B/shard%d'%i for i in range(4)]
    members=[folder+'/'+name for folder in folders for name in ('predictions.jsonl','complete.json','protocol.json')]
    dest=ROOT/'delivery/qwen35_quant_five_benchmarks'
    dest.mkdir(parents=True,exist_ok=True)
    archive=dest/'saved_outputs.tar.gz'
    with archive.open('wb') as out:
        proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
            'timeout -k 5s 90s tar -czf - -C '+REMOTE+' '+' '.join(members)],
            stdout=out,stderr=subprocess.PIPE,timeout=110)
    assert proc.returncode==0,proc.stderr.decode('utf-8','replace')
    raw=[]
    with tarfile.open(archive,'r:gz') as tar:
        assert set(tar.getnames())==set(members)
        for folder in folders:
            complete=json.load(tar.extractfile(folder+'/complete.json'))
            assert complete['generation_complete'] and complete['failures']=={'ok':complete['records']}
            rows=[json.loads(line) for line in tar.extractfile(folder+'/predictions.jsonl')]
            assert len(rows)==complete['records']
            raw.extend(r for r in rows if r['arm'] in ('cache_h8','cache_h4'))
    assert len(raw)==14472 and len({(r['arm'],r['id']) for r in raw})==14472
    judge_rows=[json.loads(line) for line in (ROOT/'delivery/qwen35_quant_judge/judge_decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(judge_rows)==3972
    judge={(r['arm'],r['id']):r for r in judge_rows}
    assert len(judge)==3972
    official=json.loads((ROOT/'delivery/summary.json').read_text(encoding='utf-8'))
    result={}
    for arm in ('cache_h16','cache_h8','cache_h4'):
        data=official['table']['Qwen3.5-9B'][arm]
        assert set(data)==set(ORDER)
        for b,n in EXPECTED.items():
            assert data[b]['complete'] and data[b]['generated']==data[b]['scored']==n
        result[arm]={b:data[b]['score'] for b in ORDER}
        result[arm]['avg']=sum(result[arm][b] for b in ORDER)/5
    for arm in ('cache_h8','cache_h4'):
        for benchmark,n in EXPECTED.items():
            rows=[r for r in raw if r['arm']==arm and r['benchmark']==benchmark]
            assert len(rows)==n and all(r['status']=='ok' for r in rows)
            cells=collections.defaultdict(list)
            for r in rows:
                score=r['score']
                if benchmark=='locomo':
                    vote=judge[(arm,r['id'])]
                    assert vote['model']=='gpt-6-astra' and vote['protocol_id']=='midcache-locomo-gpt6-astra-v1'
                    assert vote['judge_correct'] in (0,1)
                    score=vote['judge_correct']
                assert isinstance(score,(float,int)) and math.isfinite(score)
                cells[(r['task'],r['length'])].append(score)
            if benchmark=='locomo':score=100*sum(sum(v) for v in cells.values())/n
            else:
                assert len(cells)==dict(ruler=15,longeval=5,longbench=6,babilong=21)[benchmark]
                score=100*sum(sum(v)/len(v) for v in cells.values())/len(cells)
            assert math.isclose(score,result[arm][benchmark],abs_tol=1e-10)
    report=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),model='Qwen3.5-9B',
        j=6,adapter='rank32 alpha32 final4000',expected_per_arm=EXPECTED,
        scores=result,predictions=14472,judge_records=3972,errors=0,
        remote_scorer_recomputed=official['cpu_rescored'],
        collection_actual_returncode=proc.returncode,local_aggregation_verified=True,
        source_archive=str(archive),note='Remote summary recomputed four-benchmark per-item scores; local verification independently checked complete denominators and cell-macro aggregates with saved judgments.')
    (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
