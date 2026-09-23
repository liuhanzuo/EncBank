import json,statistics,datetime
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parent
def load(name):p=R/name;return json.loads(p.read_text()) if p.exists() else None
data=load('trace_results.json') or [];streams=load('stream_results.json') or [];groups=defaultdict(list)
for r in data:groups[r['variant']].append(r)
summary=[]
for name,rows in groups.items():
    counts={k:sum(r.get(k,0) for r in rows) for k in ['requested_chunks','hit_chunks','rebuilt_chunks','promoted_hits','promotions','evictions']}
    full=[r for r in rows if r['events'][0]['selected_chunks']==12]
    summary.append(dict(variant=name,measurements=len(rows),mean_prefill_ms=1000*statistics.mean(r['prefill_seconds'] for r in rows),
        mean_ttft_ms=1000*statistics.mean(r['ttft_seconds'] for r in rows),
        full12_mean_prefill_ms=1000*statistics.mean(r['prefill_seconds'] for r in full) if full else None,
        full12_measurements=len(full),hit_fraction=counts['hit_chunks']/max(1,counts['requested_chunks']),
        max_cache_mib=max(r['cache_bytes'] for r in rows)/2**20,max_peak_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30,
        mean_first_token_kl=statistics.mean(r['next_token_kl_vs_cold'] for r in rows if r['next_token_kl_vs_cold'] is not None),
        max_first_token_kl=max(r['next_token_kl_vs_cold'] or 0 for r in rows),**counts))
result=dict(at=datetime.datetime.now().astimezone().isoformat(),complete=load('complete.json'),parent=load('parent_exit.json'),
    qualification=load('qualification.json'),summary=summary,streaming=streams)
(R/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
lines=['# Hot KV buffer 初步实验','',f"快照：{result['at']}。GPU 完成记录：{result['complete']}。",'',
    'Qwen3-8B + 原 Encbank LoRA，512-token chunk，top12，GPU LRU 容量 12/24/48 个 chunk。选取六个真实 Terminal-Bench 会话最早的至多六次已返回请求，共 30 条；每组两轮独立缓存会话，第二轮反转顺序。每个请求强制一个 token 以计量 prefill，以下不是完整任务的加速率或得分。','',
    '| 配置 | 次数 | prefill 平均 ms | TTFT 平均 ms | 12块 prefill ms | KV命中率 | 当前/近期chunk命中 | 重建chunk | 缓存峰值 MiB | 分配峰值 GiB |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:
    full=f"{r['full12_mean_prefill_ms']:.2f}" if r['full12_mean_prefill_ms'] is not None else '—'
    lines.append(f"| {r['variant']} | {r['measurements']} | {r['mean_prefill_ms']:.2f} | {r['mean_ttft_ms']:.2f} | {full} | {100*r['hit_fraction']:.1f}% | {r['promoted_hits']} | {r['rebuilt_chunks']} | {r['max_cache_mib']:.1f} | {r['max_peak_gib']:.2f} |")
lines+=['','cold36 使用原联合 memory/query prefill；cold24 使用 n24 分拆路径。exact36 只接受相同有序前缀，允许 BF16 执行形状带来的小数值差异。hot36/hot24 是按 chunk 复用的近似方案：旧上下文产生的 KV 经 RoPE 位置平移后复用；当前 query/decode 已计算出的完整 chunk 主动入缓存。no_promote 关闭这种主动入缓存。','',
    '12块子集只有5条独立请求、各测两次，并包含缓存填充过程。24与48容量组实际都只驻留到17个chunk，均无淘汰，命中结果完全相同；它们的总体时间差不能归因于容量提升，两轮采样存在测量波动。','',
    '## 连续生成 1,056 tokens','',
    '固定相同输入和相同强制输出，跨过 512、1024 两个生成边界。此项是缓存与流式机制测试，不是任务质量测试。','',
    '| 配置 | 总秒数 | prefill秒数 | decode秒数 | 命中/请求chunk | 当前chunk命中 | 重建chunk | 边界 |',
    '|---|---:|---:|---:|---:|---:|---:|---|']
for r in streams:
    lines.append(f"| {r['variant']} | {r['request_seconds']:.2f} | {r['prefill_seconds']:.3f} | {r['decode_seconds']:.2f} | {r.get('hit_chunks',0)}/{r.get('requested_chunks',0)} | {r.get('promoted_hits',0)} | {r.get('rebuilt_chunks',0)} | {[e.get('generated_boundary') for e in r['events'][1:]]} |")
lines+=['','## 输出差异诊断','',
    '这里只检查每次调用第一个 token 的分布，JSON 开头通常很固定，不能据此证明任务质量保持。严格模式的依赖检查、主动入缓存收益和近似模式的质量是不同问题。完整 Terminal-Bench 需重新运行并用原判分器评估。','',
    '| 配置 | 首token KL平均 | 首token KL最大 |','|---|---:|---:|']
for r in summary:lines.append(f"| {r['variant']} | {r['mean_first_token_kl']:.6f} | {r['max_first_token_kl']:.6f} |")
(R/'READOUT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(complete=bool(result['complete']),summary=summary,stream_variants=len(streams)),indent=2))
