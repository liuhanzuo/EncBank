"""Render only observed, complete results; an unfinished cost run is explicit."""
from pathlib import Path
import json,shutil
HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]/'Encbank/paper_iclr2027_rewrite_20260912'
NAMES={'replay':'Replay j=0','w0':'Encbank j=12, w=0','w32':'Encbank j=12, w=32'}
CELLS=('longeval_8k','longeval_16k','longeval_32k','qasper')
def main():
    q=json.loads((HERE/'quality_summary.json').read_text(encoding='utf-8'))
    lines=['# Encbank 固定 w=32 独立补测','',
        '质量结果已完成：LongEval 8k/16k/32k各100例，Qasper全部200例，共500例、1500个输出。三臂共享作者确认的主实验adapter、检索IDs和query。所有逐例分数和解码文本已由独立CPU脚本复核。',
        '', '本轮w=32未按新样本调参。显式BOS=151643；历史诊断在tokenizer未设BOS时使用输入首token。新结果是独立队列，不与旧表作逐例差。plain-text greedy，LongEval上限16、Qasper128 token；本次所有输出均达到该上限。',
        '', '| 方法 | LongEval 8k Acc | 16k Acc | 32k Acc | Qasper F1 |','| --- | ---: | ---: | ---: | ---: |']
    for arm in NAMES:
        lines.append('| '+NAMES[arm]+' | '+' | '.join(f"{q['cells'][c]['scores_percent'][arm]:.2f}" for c in CELLS)+' |')
    lines+=['', '单位均为百分制。w32−w0 的95%配对bootstrap区间（20,000次，以例为重采样单位，未作多重比较校正）：']
    for c in (*CELLS,'longeval_macro'):
        d=q['cells'][c]['w32_minus_w0'];a,b=d['ci95_pp']
        lines.append(f"- {c}: {d['difference_pp']:+.2f} 分，[{a:.2f}, {b:.2f}]。")
    lines+=['', 'LongEval三长度macro为77.67/72.67/62.33；w32相对w0有28例改对、59例改错。Qasper w32相对w0的区间跨零，且绝对F1低，不能写成通用QA能力得到保持。',
        '', '探索性边界检查：目标记录始于chunk前32个位置的20例从45升至80；不跨边界的290例从74.48降至63.10。两个子集并不互斥，前者样本小，不能据此完成机制归因。总结果否定了本协议下通用修复的说法。',
        '', '## RTX5090 含Write成本']
    cp=HERE/'cost_summary.json'
    if not cp.exists():
        lines+=['', '**仍在运行，以下不填成本结论。** 预先固定每格前5例，共20例；每个方法、每例一次预热、三次正式重复、三个独立进程。完整文档重新Write，固定128个生成token。原始记录在cost/process_01至03，进度在各目录progress.json。']
    else:
        c=json.loads(cp.read_text(encoding='utf-8'));assert c['complete']
        lines+=['', '三个独立进程均完成，共540次正式请求（另有180次预热），质量样本仍只有20例，不把时间重复当独立质量样本。以下为进程内五个文档×三次重复的中位数，再取三个进程中位数；GPU取正式请求最大值。单位ms，GPU为十进制GB。',
            '', '| 输入 | 方法 | 本机质量 | 文档Write | TTFT | 在线含128输出 | E2E含Write | 峰值GPU |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for cell in CELLS:
            for arm in NAMES:
                r=c['cells'][cell][arm]
                if r['status']!='ok':
                    lines.append(f'| {cell} | {NAMES[arm]} | OOM | OOM | OOM | OOM | OOM | OOM |');continue
                values=[f"{r['local_quality_percent']:.2f}"]+[f"{r[m]['median']:.1f}" for m in ('document_write_ms','ttft_ms','online_ms','e2e_ms')]+[f"{r['peak_allocated_GB']:.2f}"]
                lines.append('| '+cell+' | '+NAMES[arm]+' | '+' | '.join(values)+' |')
        lines+=['','w32/w0 实测比值（各方法先独立聚合；不是逐次配对差的区间）：']
        for cell in CELLS:
            ratios=c['cells'][cell].get('w32_over_w0',{})
            if ratios:lines.append('- '+cell+': '+', '.join(f'{m} {v:.3f}×' for m,v in ratios.items())+'。')
        disagreement={cell:{a:c['cells'][cell][a].get('score_disagreement_items') for a in NAMES} for cell in CELLS}
        lines+=['','不同重复间出现质量分数差异的样本数：`'+json.dumps(disagreement,ensure_ascii=False)+'`。质量按每例重复均值、再按例均值报告，详情及跨设备差异保存在cost_summary.json。','',
            '实际allocator上限约26.077GB，严于28GB总预算：run_cost将28e9/1024³传给以十进制GB为单位的gpu_gate。三个进程运行表达式相同，所有成功请求另行验证allocated/reserved均不超过28e9；无OOM时不把较紧上限作为性能优势。该澄清覆盖原metadata中的名义cap_bytes。',
            '', '只测单请求、无并发、20例小子集。TTFT排除全篇准备，E2E计入准备和文档Write；分词/模型加载/外部I/O排除。replay fetch分量还含embedding，不能仅按该列推断传输差。持久字节指残差或token-ID对象，不包含额外索引和重复token副本。w0与w32每例持久状态字节完全一致。',
            '', '该报告由完整原始记录生成。配对成本表和协议已加入论文附录B.2，正文§5.3引用其中的Write与E2E增量；表格仅使用预设20例成本子集。']
    lines+=['','## 定义核对','',
        '原contextual control会同时扩大文档块、query和下层decode的上下文；不能单独归因于document缓存。Overlap只扩文档块的Write。论文METHOD_VISIBILITY_zh.md记录原始仓库函数与teacher/adapter配置；PROTOCOL_zh.md记录实验协议。',
        '', '主表使用默认w=0；本组配对质量与成本结果在附录B.2报告。当前500例支持的结论是：multikey局部修复不推广到LongEval；Qasper没有明确改善。']
    report='\n'.join(lines)+'\n'
    (HERE/'RESULTS_zh.md').write_text(report,encoding='utf-8')
    (PAPER/'OVERLAP_CONFIRMATION_zh.md').write_text(report,encoding='utf-8')
    shutil.copy2(HERE/'quality_summary.json',PAPER/'overlap_quality_summary.json')
    if cp.exists():shutil.copy2(cp,PAPER/'overlap_cost_summary.json')
    print('Updated observed-results report; cost complete:',cp.exists())
if __name__=='__main__':main()
