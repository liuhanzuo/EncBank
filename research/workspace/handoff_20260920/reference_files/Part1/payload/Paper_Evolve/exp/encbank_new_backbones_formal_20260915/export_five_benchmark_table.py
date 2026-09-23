"""Join the two verified deliveries without changing either evaluation protocol."""
import csv, json, math, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery'
MODELS={'Qwen3.5-9B':6,'Qwen3.8-27B':21}
METHODS={'kvdirect':'KV-Direct (full source)',
         'replay_base':'Selected-text replay',
         'replay_shared_lora':'Replay (shared Encbank LoRA)',
         'streamingllm':'StreamingLLM-style',
         'hcache_style':'HCache-style',
         'cache_without_lora':'Encbank (without LoRA)',
         'cache_lora':'Encbank'}
KEYS=['ruler','longeval','longbench','babilong','locomo','avg']
HEADERS=['RULER','LongEval','LongBench F1','BABILong','LoCoMo','Avg.']


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    first=json.loads((OUT/'four_benchmark_check.json').read_text())
    second=json.loads((OUT/'locomo_astra_verification.json').read_text())
    assert first['complete'] and second['complete']
    assert first['records']==73500 and second['records']==27804
    assert first['oom']==second['oom']==second['unscored']==0
    def read(name):
        with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        mapping={(r['model'],r['arm']):r for r in rows}
        assert len(rows)==len(mapping)==14
        return mapping
    four=read('four_benchmark_results.csv');locomo=read('locomo_astra_results.csv')
    assert four.keys()==locomo.keys()=={(m,a) for m in MODELS for a in METHODS}
    rows=[]
    for model,j in MODELS.items():
        for arm,label in METHODS.items():
            a,b=four[model,arm],locomo[model,arm]
            assert int(a['j'])==int(b['j'])==j and int(b['n'])==1986
            values={key:float(a[key]) for key in KEYS[:4]}
            values['locomo']=float(b['score_pct'])
            assert all(math.isfinite(v) and 0<=v<=100 for v in values.values())
            values['avg']=sum(values.values())/5
            rows.append(dict(model=model,j=j,arm=arm,method=label,**values))
    maxima={(m,k):max(r[k] for r in rows if r['model']==m) for m in MODELS for k in KEYS}
    def formatted(row,key,latex=False):
        value=f'{row[key]:.2f}'
        if math.isclose(row[key],maxima[row['model'],key],abs_tol=1e-9):
            return '\\textbf{'+value+'}' if latex else '**'+value+'**'
        return value
    lines=['# Qwen3.5-9B 与 Qwen3.8-27B：五项 benchmark 汇总','',
        '按论文预定任务与长度范围，两个模型的五项 benchmark、七种方法均已完成并核验。共 101,304 条预测，0 OOM、0 未评分。9B 的缓存深度 j=6，27B 为 j=21；两模型的 Encbank adapter 分别训练 4,000 步。','',
        '所有数值为 0–100，越高越好。Avg. 是五项分数的描述性等权平均，不代表同一量纲的统计估计。加粗为同一模型内的列最大值。','',
        '| 模型 | 方法 | '+' | '.join(HEADERS)+' |',
        '|---|---|'+'---:|'*6]
    for row in rows:
        lines.append('| '+row['model']+' | '+row['method']+' | '+' | '.join(formatted(row,k) for k in KEYS)+' |')
    lines+=['','## 评测范围与解释','',
        '- RULER：3 个任务 × 5 个长度（8k–128k），每单元100题，共1500题，15单元宏平均。',
        '- LongEval：8k/16k/32k/64k/128k，各100题，共500题，五长度宏平均。',
        '- LongBench：六个 QA 数据集，共1150题，数据集 F1 等权平均。',
        '- BABILong：qa1/qa2/qa5 × 7 个长度（0k–32k），每单元100题，共2100题，21单元宏平均。',
        '- LoCoMo：全部1986题；1540道类别1–4问题由同一 gpt-6-astra/low 固定盲评 prompt 评分，446道类别5问题按原本地拒答规则评分，按题数加权。不是仅1540题的语义小计，也不是另一个H16/H8/H4精度实验的Astra/high协议。服务别名的底层日期快照未经核实。',
        '', '两个模型采用各自原生 chat 模板并关闭 thinking。所有方法共享样本、输出预算与评分口径；LongEval 最多输出48 tokens。这与原 Qwen3-8B 主表的 plain-text 协议不同，应作为新模型扩展表单独呈现。',
        '', 'Selected-text replay 重放与 Encbank 相同的检索文本；shared LoRA 是复用 Encbank adapter，并非单独训练 replay。KV-Direct 使用完整 source。StreamingLLM-style 和 HCache-style 是兼容参考实现。InfLLM 与 MemoryLLM 未纳入这两种 backbone 的本批比较；没有其兼容运行结果，不作填值。本表也不包含量化 H8/H4 配置。',
        '', 'Encbank 的 LoRA 在两模型的五项指标上均改善 without-LoRA 分数，但并非各项都优于其他方法。27B 的 BABILong 为本模型组最高；两模型的五项等权均值均低于完整 source 的 KV-Direct。这些是单训练种子的质量结果，不据此推导显著性或速度优势。',
        '', '原始汇总来源：`four_benchmark_results.csv`、`locomo_astra_results.csv`。原始生成及评分证据仍保存在各自交付包中。`five_benchmark_table.tex` 可供论文直接引用；本脚本不修改当前稿件。']
    (OUT/'FIVE_BENCHMARK_RESULTS_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tex=[r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',
        r'\caption{\textbf{Five-benchmark quality on two additional backbones.} Encbank uses $j=6$ for Qwen3.5-9B and $j=21$ for Qwen3.8-27B. Scores are 0--100 ($\uparrow$); Avg. is their descriptive unweighted mean. Bold marks the largest value within each backbone.}',
        r'\label{tab:new-backbones}',r'\begin{tabular}{llrrrrrr}',r'\toprule',
        r'Backbone & Method & RULER & LongEval & LongBench & BABILong & LoCoMo & Avg.\\',r'\midrule']
    for i,row in enumerate(rows):
        if i==7:tex.append(r'\midrule')
        tex.append(row['model']+' & '+row['method']+' & '+' & '.join(formatted(row,k,True) for k in KEYS)+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\par\vspace{3pt}',r'\begin{minipage}{\textwidth}\footnotesize',
        r'RULER: 15 task--length cells; LongEval: five lengths, 8k--128k; LongBench: six-dataset F1 macro; BABILong: three tasks over seven lengths. LoCoMo averages all 1,986 questions: a common GPT-6-Astra judge for categories 1--4 and the original local abstention rule for category 5. Both backbones use native chat templates with thinking disabled; these results are separate from the Qwen3-8B plain-text protocol. Shared LoRA means the Encbank adapter reused for replay. StreamingLLM-style and HCache-style are compatible reference implementations. InfLLM and MemoryLLM were not evaluated on these backbones.',
        r'\end{minipage}',r'\end{table*}']
    (OUT/'five_benchmark_table.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    (OUT/'five_benchmark_results.json').write_text(json.dumps(dict(complete=True,records=101304,rows=rows),indent=2)+'\n')
    print('\n'.join(lines[:23]))


if __name__=='__main__':main()
