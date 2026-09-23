import json,math
from pathlib import Path
from queue_metrics import summarize
ROOT=Path(__file__).resolve().parent;D=ROOT/'delivery'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
    wait=load(D/'parent_exit.json');assert wait['actual_wait'] and wait['returncode']==0
    assert load(D/'correctness.json')['passed']
    points=[]
    for p in sorted((D/'results').iterdir()):
        if not p.is_dir():continue
        r=load(p/'complete.json');parts=p.name.split('_');k=int(parts[0][1:]);c=int(parts[-1][1:]);method='_'.join(parts[1:-1])
        if r['status']=='ok':
            rows=[json.loads(x) for x in (p/'requests.jsonl').read_text().splitlines()]
            assert len(rows)==len({x['id'] for x in rows})==64
            for row in rows:
                assert row['arrival']<=row['start']<=row['first']<=row['end']
                assert all(len(s)==k for s in row['detail']['selected'])
                assert all(len(ids)==32 for ids in row['detail']['generated_ids'])
            check=summarize(rows,max(x['end'] for x in rows),32)
            for key in ['requests_per_s','output_tokens_per_s','wall_s']:assert math.isclose(check[key],r[key],rel_tol=1e-9)
            for key in ['ttft_s','e2e_s','queue_s','service_s']:assert check[key]==r[key]
            r['offline']=load(p/'offline.json')
        r.update(point=p.name,k=k,method=method,concurrency=c);points.append(r)
    assert len(points)==24
    summary=dict(complete=True,points=points,environment=load(D/'environment.json'),agent_task_success_measured=False)
    summary['metric_units']=dict(point_percentile_fields=['queue_s','ttft_s','e2e_s','service_s'],point_percentile_unit='milliseconds despite legacy _s keys',raw_request_duration_unit='seconds',raw_request_timestamp_unit='seconds',wall_s_unit='seconds')
    summary['allocator_cap_bytes']=128*2**30
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    text='# Qwen3-8B 检索预算：成本扫描\n\n'
    text+='固定j12和原adapter，k12/24/32/48；BM25 hop2。3份32k源，512 query，固定32输出；每点16预热+64正式请求，单次探索，不报告稳定p99。这是系统成本，不是Agent成功率或质量结论。GPU入口由用户指定B300，原始型号/架构/显存见summary.json。\n\n'
    text+='本扫描统一限制PyTorch allocator为128 GiB，低于整卡267.69 GiB物理显存；表内OOM表示在这一实验额度内不可运行，不等于整卡容量上限。失败阶段与申请量保留在各点complete.json。\n\n'
    for c in [1,4,16]:
        text+=f'## 并发{c}\n\n| k | 历史位置 | 方法 | 输出tokens/s | req/s | TTFT p50/p95 ms | E2E p95 ms | allocated/reserved GiB | NVML峰值 GiB |\n|---:|---:|---|---:|---:|---:|---:|---:|---:|\n'
        for k in [12,24,32,48]:
            for method in ['raw','h_gpu']:
                r=next(x for x in points if x['k']==k and x['concurrency']==c and x['method']==method)
                name='文本回放' if method=='raw' else 'GPU Hidden'
                if r['status']!='ok':text+=f'| {k} | {k*512} | {name} | OOM | — | — | — | — | — |\n';continue
                text+=f"| {k} | {k*512} | {name} | {r['output_tokens_per_s']:.2f} | {r['requests_per_s']:.3f} | {r['ttft_s']['p50']:.2f}/{r['ttft_s']['p95']:.2f} | {r['e2e_s']['p95']:.2f} | {r['peak_allocated_bytes']/2**30:.3f}/{r['peak_reserved_bytes']/2**30:.3f} | {r['nvml_sampled_peak_bytes']/2**30:.3f} |\n"
    text+='\n后缀prefill与活动KV随读取预算增长；持久hidden bank内容相同。相同Transformers/CoMem参考执行器与同步微批，不是原生vLLM性能；NVML为100ms采样，精确allocator峰值另列。离线Write与索引不计入在线窗口；其耗时在各点offline.json。\n'
    text+='\nAgent默认预算仍需代码/日志闭环的任务成功率验证。现有训练主要使用7个历史块+查询块，不能从更大的上下文容量推断更高质量。\n'
    (ROOT/'RESULTS_zh.md').write_text(text,encoding='utf-8')
    print(str(ROOT/'RESULTS_zh.md'))
if __name__=='__main__':main()
