"""Summarize completed server measurements; no model calls."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
out=ROOT/'results'
summary=json.loads((out/'summary.json').read_text())
receipt=json.loads((out/'parent_exit.json').read_text())
validation=json.loads((out/'validation.json').read_text())
assert summary['complete'] and receipt['actual_wait'] and receipt['returncode']==0
assert validation['passed'] and len(validation['checks'])==3
assert summary['formal_observations']==378
assert all(c['n']==21 for c in summary['cells'])
environment=summary['environment']
decode=summary['decode_512_ms']['median']
names={'exact_prefix':'保留有序前缀', 'independent_serial':'独立块逐块重建', 'independent_batch':'独立块合批重建'}
cells={(c['mode'],c['misses']):c for c in summary['cells']}
lines=['# KV 重建计时结果','',
    f"作业 {receipt['job']}，主机 {environment['host']}，GPU {environment['gpu']}（实际驱动名称），计算能力 {environment['cc']}。",
    'Qwen3-8B，j=12，BF16 骨干、原 FP32 unmerged rank32 LoRA；512-token 历史块，固定 12 块活动历史，加一个 sink。',
    '三个保存的 PG19 输入，每输入每格 2 次预热＋7 次正式测量，共 378 条正式计时；单模型进程。所有推理均在服务器执行。','',
    '## 核心结果：纯重建时间（ms，中位数）','',
    '| 未命中块数 | 保留有序前缀 | 独立块逐块重建 | 独立块合批重建 |',
    '|---:|---:|---:|---:|']
for m in [0,1,2,4,8,12]:
    vals=[cells[(mode,m)]['rebuild_wall_ms']['median'] for mode in names]
    lines.append('| '+str(m)+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
lines += ['', '纯重建：输入 H 已在 GPU，运行第 13–36 层并保留 KV。不含初次 Write、检索、查询编码、LM head、缓存恢复、持久块复制、活动 KV 组装或 decode。0 未命中时不执行重建，记为 0。', '',
    '## 含实际缓存复制的总时间（ms，中位数）','',
    '| 未命中块数 | 保留有序前缀 | 独立块逐块重建 | 独立块合批重建 | 合批总时间 / 512-token decode |',
    '|---:|---:|---:|---:|---:|']
for m in [0,1,2,4,8,12]:
    vals=[cells[(mode,m)]['total_wall_ms']['median'] for mode in names]
    lines.append('| '+str(m)+' | '+' | '.join(f'{x:.3f}' for x in vals)+f' | {100*vals[-1]/decode:.3f}% |')
lines += ['', '前缀方案包括已有前缀 KV 的实际 clone；独立块方案包括新块 clone 成独立持久 storage，并与命中块真实拼接为活动 KV。总时间还含 Python 调度和分阶段同步，因此不能只把各阶段中位数相加。全命中仍有活动 KV 恢复/组装开销。', '',
    '## 生成参照','',
    f"当前联合文档 attention 下，输出 512 tokens（511 次 decode forward）的中位耗时为 **{decode/1000:.3f} 秒**，三个输入范围 {summary['decode_512_ms']['min']/1000:.3f}–{summary['decode_512_ms']['max']/1000:.3f} 秒。",
    f"每次 decode forward 中位耗时 **{summary['decode_ms_per_token']['median']:.3f} ms**。query prefill 单列于 decode.json。固定输出长度、忽略 EOS，仅测系统成本。",
    '上表百分比是单次缓存刷新相对于该固定长度生成参照的比值，不是实测动态检索 agent 的端到端减速率。','',
    '## 边界','',
    '- 保留有序前缀：相同选块顺序和位置，已有 KV 是连续前缀，未命中的是后缀。',
    '- 独立块逐块/合批：每块独立 attention、局部位置；二者计算图一致，但与原 Encbank 的跨块文档 attention 不同。本次不评价回答质量。',
    '- 未实现真实每 512 tokens 动态检索、LRU 淘汰或跨调用的命中轨迹；本次固定设置未命中数量，以分离核心成本。',
    '- 缓存验证检查形状、有限值、每块 48 MiB KV、前缀重建与完整重建、独立块串行与合批的数值差异。逐项差异保存在 validation.json，BF16 相对 RMS 阈值为 3%，不宣称 bitwise 一致。',
    '- 未测 H24 双深度重建，也未测 27B；当前数字仅适用于上述 8B 配置。',
    '- 单 GPU、单进程、三个输入；重复测量不是独立的生产负载试验，不把本次 p95 当成稳定尾延迟结论。','',
    '原始记录：measurements.jsonl；汇总：summary.json；生成记录：decode.json；环境：environment.json；真实进程退出：parent_exit.json。','']
(out/'RESULTS_zh.md').write_text('\n'.join(lines),encoding='utf-8')
with (out/'timings.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f)
    writer.writerow(['mode','misses','rebuild_ms','restore_ms','store_ms','assembly_ms','total_ms','pct_decode512'])
    for mode in names:
        for m in [0,1,2,4,8,12]:
            c=cells[(mode,m)]
            writer.writerow([mode,m]+[c[k]['median'] for k in ['rebuild_wall_ms','restore_wall_ms','store_wall_ms','assembly_wall_ms','total_wall_ms']]+[100*c['total_wall_ms']['median']/decode])
print(json.dumps(dict(complete=True,report=str(out/'RESULTS_zh.md'),decode_512_seconds=decode/1000),ensure_ascii=False))
