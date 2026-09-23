"""Recompute completed serving metrics from real requests; no inferred speedups."""
import json,math,statistics
from pathlib import Path
from queue_metrics import summarize
ROOT=Path(__file__).resolve().parent;D=ROOT/'delivery'
NAMES=dict(raw='文本回放',raw_kv='文本回放 + 前缀KV',h_cpu='CPU Hidden',h_gpu='GPU Hidden',hybrid='GPU Hidden + 上层KV')
ORDER=list(NAMES)
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    wait=load(D/'parent_exit.json');assert wait['actual_wait'] and wait['returncode']==0
    assert load(D/'correctness.json')['passed']
    metadata=load(D/'environment.json');points=[]
    for path in sorted((D/'results').glob('*')):
        r=load(path/'complete.json');p=path.name
        if r['status']!='ok':points.append(dict(point=p,**r));continue
        rows=[json.loads(s) for s in (path/'requests.jsonl').read_text().splitlines()]
        assert len(rows)==len({x['id'] for x in rows})==256
        for row in rows:
            assert 0<=row['arrival']<=row['start']<=row['first']<=row['end']
            assert all(len(ids)==32 for ids in row['detail']['generated_ids'])
            assert math.isclose(row['ttft_s'],row['first']-row['arrival'],abs_tol=1e-6)
            assert math.isclose(row['e2e_s'],row['end']-row['arrival'],abs_tol=1e-6)
            assert math.isclose(row['end']-row['first'],row['detail']['token_times'][-1]-row['detail']['token_times'][0],abs_tol=1e-6)
        check=summarize(rows,max(x['end'] for x in rows),32)
        for key in ['requests_per_s','output_tokens_per_s','wall_s']:assert math.isclose(check[key],r[key],rel_tol=1e-9)
        for key in ['ttft_s','e2e_s','queue_s','service_s']:assert check[key]==r[key]
        assert r['persistent_gpu_H_bytes']+r['persistent_gpu_KV_bytes']<=8*2**30
        r.update(point=p,offline=load(path/'offline.json'))
        r['prefix_hit_fraction']=r['prefix_stats']['matched_positions']/r['prefix_stats']['prefix_positions']
        points.append(r)
    assert len(points)==30
    summary=dict(complete=True,environment=metadata,points=points,successful_points=sum(r['status']=='ok' for r in points),
                 source_length=32768,selected_tokens=6144,query_tokens=512,output_tokens=32,adapter_shared=True,
                 native_vllm=False,continuous_batching=False,persistent_budget_bytes=8*2**30)
    summary['metric_units']=dict(point_percentile_fields=['queue_s','ttft_s','e2e_s','service_s'],point_percentile_unit='milliseconds despite legacy _s keys',raw_request_duration_unit='seconds',raw_request_timestamp_unit='seconds',wall_s_unit='seconds')
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text='# 单卡 B300：Qwen3-8B 缓存部署原型结果\n\n'
    text+='同一模型/原j12 adapter、同一参考执行实现；驱动显示 '+metadata['name']+'，CUDA sm_103，设备UUID '+metadata['gpu_uuid']+'。不是原生vLLM或PagedAttention性能。固定32k源文档、12×512检索块、512查询tokens和32输出tokens。每点16预热+256正式请求，单次探索测量；报告p50/p95，不把p99当稳定结论。\n\n'
    text+='五方案共享8GiB持久GPU缓存预算，H和热点KV共同计入；活动KV另计入128GiB统一allocator上限。CPU Hidden通过预分配pinned缓冲区异步搬运。前缀KV按完整有序前缀精确匹配，叶节点LRU淘汰；不复用问题或答案。同步微批，按命中长度分组prefill；恢复热点KV和合并活跃状态仍有真实拷贝。\n\n'
    correctness=load(D/'correctness.json')
    text+='数值检查采用BF16容差而非bitwise相等。完整检查见delivery/correctness.json；首次hybrid全命中诊断在24个相同历史位置中有1个argmax变化，最大logits差0.4375、最大RMS0.14243，小于未缓存batch/scalar变化幅度。自由生成差异已保留，本报告只比较固定32-token系统成本，不据此宣称质量不变。\n\n'
    for profile in ['hot','diverse']:
        text+='## '+('热门：8个检索前缀' if profile=='hot' else '多样：96查询、84个检索前缀')+'\n\n'
        text+='| 方法 | 并发 | req/s | 输出tokens/s | TTFT p50/p95 ms | E2E p95 ms | allocated/reserved GiB | NVML峰值 GiB | 前缀位置命中率 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for method in ORDER:
            for c in [1,4,16]:
                r=next(x for x in points if x['point']==f'{profile}_{method}_c{c}')
                if r['status']!='ok':text+=f'| {NAMES[method]} | {c} | OOM | — | — | — | — | — | — |\n';continue
                text+=f"| {NAMES[method]} | {c} | {r['requests_per_s']:.3f} | {r['output_tokens_per_s']:.2f} | {r['ttft_s']['p50']:.2f}/{r['ttft_s']['p95']:.2f} | {r['e2e_s']['p95']:.2f} | {r['peak_allocated_bytes']/2**30:.3f}/{r['peak_reserved_bytes']/2**30:.3f} | {r['nvml_sampled_peak_bytes']/2**30:.3f} | {r['prefix_hit_fraction']:.1%} |\n"
    text+='\n命中率为正式请求中命中的文档前缀位置/6145（含sink）；hidden本身的文档命中不计作KV命中。GPU峰值包含模型、持久状态、临时工作区及活动KV；NVML为100ms采样设备占用，不能视为精确瞬时峰值。离线Write、索引准备与CPU RSS、缓存实际字节、TPOT/ITL、全部逐请求时戳见summary.json和delivery/results。TTFT/E2E从提交至首/末CPU可用token，含排队/在线检索/搬运/缓存维护/模型执行，不含HTTP网络与token化。吞吐以完整正式窗口为分母。\n'
    text+='\n## 并发16下的配对比较\n\n'
    text+='以下为同一负载的单次观测差异，尚无重复运行置信区间；这些性能点不证明不同算法的回答质量相同。\n\n'
    text+='| 访问模式 | 比较（候选 / 参考） | 输出吞吐比 | allocated峰值比 | TTFT p95比 |\n|---|---|---:|---:|---:|\n'
    for profile in ['hot','diverse']:
        subset={r['method']:r for r in points if r['status']=='ok' and r['profile']==profile and r['concurrency']==16}
        for candidate,reference in [('h_gpu','raw'),('hybrid','raw_kv'),('h_cpu','h_gpu'),('hybrid','h_gpu')]:
            if candidate not in subset or reference not in subset:continue
            a,b=subset[candidate],subset[reference]
            text+=f"| {profile} | {NAMES[candidate]} / {NAMES[reference]} | {a['output_tokens_per_s']/b['output_tokens_per_s']:.3f} | {a['peak_allocated_bytes']/b['peak_allocated_bytes']:.3f} | {a['ttft_s']['p95']/b['ttft_s']['p95']:.3f} |\n"
    text+='\n吞吐比大于1表示候选更快；显存比或延迟比小于1表示候选较低。CPU Hidden 的主机内存单列，不能把显存迁出解释为总存储成本消失。离线Write与索引构建未包含于在线吞吐。\n'
    text+='\n测试口径参考 [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) 的并发扫描与缓存复用负载，以及 [NVIDIA NIM metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) 的吞吐和延迟定义。本轮共用参考执行器，不能直接作为与原生vLLM的引擎性能比较。\n'
    (ROOT/'RESULTS_zh.md').write_text(text,encoding='utf-8');print(json.dumps(dict(report=str(ROOT/'RESULTS_zh.md'),successful_points=summary['successful_points'])))
if __name__=='__main__':main()
