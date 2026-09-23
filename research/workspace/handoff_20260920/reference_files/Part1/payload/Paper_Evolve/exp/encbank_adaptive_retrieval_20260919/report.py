import json,math
from pathlib import Path
from queue_metrics import percentile
ROOT=Path(__file__).resolve().parent;D=ROOT/'delivery'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    wait=load(D/'parent_exit.json');assert wait['actual_wait'] and wait['returncode']==0
    assert load(D/'correctness.json')['passed']
    points=[]
    for d in sorted((D/'results').iterdir()):
        if not d.is_dir():continue
        r=load(d/'complete.json')
        if r['status']=='ok':
            rows=[json.loads(x) for x in (d/'requests.jsonl').read_text().splitlines()]
            assert len(rows)==12 and len({x['id'] for x in rows})==12
            for row in rows:
                assert len(row['generated_ids'])==len(row['token_times'])==32
                assert 0<len(row['selected'])==row['n_chunks']<=48
                assert row['selected']==sorted(set(row['selected']))
                assert math.isclose(row['ttft_ms'],1000*(row['token_times'][0]-row['start_time']),abs_tol=1e-5)
                assert math.isclose(row['e2e_ms'],1000*(row['token_times'][-1]-row['start_time']),abs_tol=1e-5)
                assert math.isclose(row['selector_ms'],1000*(row['after_selection']-row['after_query_write']),abs_tol=1e-5)
            assert r['mean_chunks']==sum(x['n_chunks'] for x in rows)/12
            for key in ['query_write_ms','selector_ms','ttft_ms','e2e_ms']:
                for p in [50,95]:assert r[key][f'p{p}']==percentile([x[key] for x in rows],p)
        points.append(r)
    assert len(points)==16
    summary=dict(complete=True,points=points,environment=load(D/'environment.json'),agent_task_success_measured=False,quality_measured=False)
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    text='# k48与top-p选择器：工程诊断\n\n'
    text+='此处是12个固定PG19查询上的选择数量和成本，不是闭环Agent或质量结果。仍使用原j12 adapter；候选上限48块、每块512位置，query最后32位置用于Q/K探针，固定32输出。所有方案单并发、共享GPU hidden bank。\n\n'
    text+='| 分支 | 平均块数（最小/最大） | selector p50 ms | TTFT p50 ms | E2E p50 ms | allocated峰值 GiB | cap不足p的请求数 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in points:
        if r['status']!='ok':text+=f"| {r['arm']} | OOM | — | — | — | — | — |\n";continue
        text+=f"| {r['arm']} | {r['mean_chunks']:.2f} ({r['min_chunks']}/{r['max_chunks']}) | {r['selector_ms']['p50']:.2f} | {r['ttft_ms']['p50']:.2f} | {r['e2e_ms']['p50']:.2f} | {r['peak_allocated_bytes']/2**30:.3f} | {r['cap_limited_requests']}/12 |\n"
    text+='\nBM25 p指非负分数归一化后的累计占比，不是相关性概率。iter48开头表示条件于同一48候选，qk表示首Read层实际Q/K投影及RoPE、跨head/query平均的记忆条件attention质量。候选之外的遗漏、不同head所需证据和重新打包的位置变化，不能由此p保证。Q/K k48是额外打分但不裁剪的开销对照。\n'
    text+='\n默认Agent试用按用户要求先设k48，再对照这些自适应策略；须用实际修复成功率验证，不能凭块数减少宣布更优。旧reader_attn仅为hidden cosine，和这里的qk不同。GPU数值检查见delivery/correctness.json，逐请求质量与时间记录见delivery/results。此处generated_ids用于记录实际解码工作，不提供任务质量分数。\n'
    text+='\n参考[Twilight论文](https://papers.nips.cc/paper_files/paper/2025/file/ad2f57116d62a9d8dcfaba6a168715e4-Paper-Conference.pdf)的先选择后top-p裁剪；本实现是Encbank块级检索变体，不是原生Twilight复现。\n'
    (ROOT/'RESULTS_zh.md').write_text(text,encoding='utf-8');print(str(ROOT/'RESULTS_zh.md'))
if __name__=='__main__':main()
