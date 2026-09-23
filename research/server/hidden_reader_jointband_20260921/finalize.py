import hashlib,json,statistics,subprocess,sys,tarfile
from pathlib import Path
from metrics import load_phase,groups
R=Path(__file__).resolve().parent
def save(name,x):(R/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
data,outputs=load_phase('confirm');confirmation=groups(data)
save('confirmation_scores.json',confirmation);save('confirmation_scored_items.json',outputs)
manifest=json.loads((R/'source_manifest.json').read_text())
for rel,expected in manifest.items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==expected,rel
jobs=[]
for phase in ['screen','confirm']:jobs+=json.loads((R/('submissions_'+phase+'.json')).read_text())
account=[]
for j in jobs:
    row=subprocess.check_output(['sacct','-X','-j',j['job'],'--format=JobID,JobName%50,State,ExitCode,Elapsed,NodeList','--noheader','-P'],text=True)
    assert 'COMPLETED|0:0' in row,(j,row)
    account.append(row)
(R/'accounting.txt').write_text(''.join(account))
runtime=[]
for j in jobs:
    if j['phase']=='screen':
        for x in json.loads((R/j['run']/'results/runtime.json').read_text()):runtime.append(dict(cell=j['cell'],**x))
runtime_summary=[]
for k in [1,4,12]:
    for n in [12,14,16,18,20,24,28,32,36]:
        selected=[x for x in runtime if x['chunks']==k and x['n']==n]
        pairs=[]
        for x in selected:
            base=next(y for y in runtime if y['cell']==x['cell'] and y['chunks']==k and y['n']==36)
            pairs.append(x['median_wall_ms']/base['median_wall_ms'])
        runtime_summary.append(dict(n=n,chunks=k,median_across9_cells_ms=statistics.median(x['median_wall_ms'] for x in selected),
            median_paired_ratio_to_n36=statistics.median(pairs),cell_medians=[x['median_wall_ms'] for x in selected]))
save('runtime_summary.json',runtime_summary)
screen=json.loads((R/'screen_scores.json').read_text());selection=json.loads((R/'selection.json').read_text())
verification=dict(source_files_unchanged=len(manifest),successful_jobs=len(jobs),all_actual_wait_zero=True,
    all_slurm_completed_zero=True,screen_items=288,confirmation_items=612,
    generations=sum(json.loads((R/j['run']/'results/summary.json').read_text())['generations'] for j in jobs),
    selected_n=selection['selected_n'],model_weights_downloaded=False)
save('verification.json',verification)
labels={'single':'单目标NIAH','multikey':'多键干扰NIAH','vt':'变量追踪（Encbank版）'}
lines=['# 联合因果attention深度实验','',
    f"本次筛选选中n={selection['selected_n']}；独立确认集覆盖612题，具体均分与逐题变化见下表。计时硬件为NVIDIA L20D。结论仅适用于本次三类任务及固定生成预算。",'',
    'Qwen3-8B + 原Encbank LoRA，冻结全部模型参数，不使用此前训练的hidden→KV头。缓存独立chunk的H12，历史第13～n层联合因果计算，第n+1～36层仅在各chunk内部计算；问题与生成token在上层仍读取全部检索历史。','',
    'n=12代表上层历史全部块内计算，n=36代表上层历史全部联合计算。每层仍执行原Attention和MLP；减少的是跨chunk的attention范围，没有省略Transformer层。原Encbank原生联合prefill另外作为native对照。','',
    '## 32题／组合的深度扫描','',
    '| n | 联合层数 | 单目标得分 | 多键得分 | 变量追踪得分 | 12块重建ms | 相对n36时间 |',
    '|---|---:|---:|---:|---:|---:|---:|']
native=[screen['tasks'][t]['score']['native'] for t in labels]
lines.append('| 原Encbank | 24 | '+' | '.join(f'{v:.2f}' for v in native)+' | — | — |')
for n in [12,14,16,18,20,24,28,32,36]:
    rt=next(x for x in runtime_summary if x['n']==n and x['chunks']==12)
    values=[screen['tasks'][t]['score']['n'+str(n)] for t in labels]
    lines.append(f'| {n} | {n-12} | '+' | '.join(f'{v:.2f}' for v in values)+f" | {rt['median_across9_cells_ms']:.2f} | {rt['median_paired_ratio_to_n36']:.3f}× |")
lines += ['',f"按预先登记的筛选规则，选中n={selection['selected_n']}：三个任务的平均损失均不超过2个百分点、任一任务/长度组合损失不超过5个百分点的最浅n。若无候选满足则回退n36；本次回退标志为{selection['fallback_used']}。这只是探索筛选条件，不是无损保证。",'',
    '## 其余68题／组合的独立确认','',
    '| 配置 | 单目标得分 | 多键得分 | 变量追踪得分 |','|---|---:|---:|---:|']
for arm in selection['confirm_arms']:
    lines.append('| '+arm+' | '+' | '.join(f"{confirmation['tasks'][t]['score'][arm]:.2f}" for t in labels)+' |')
for task,label in labels.items():
    x=confirmation['tasks'][task]['vs_native']['n'+str(selection['selected_n'])];ci=x['paired95ci_pp']
    lines += ['',f"{label}：选中配置相对原Encbank {x['delta_pp']:+.2f}个百分点，成对bootstrap95%区间[{ci[0]:+.2f}, {ci[1]:+.2f}]；{x['lower_score_items']}题得分下降，{x['higher_score_items']}题提升。"]
lines += ['','## 检索块数量与耗时','',
    '| 检索chunk数 | 选中n的重建ms | n36重建ms | 选中n相对n36时间 |',
    '|---:|---:|---:|---:|']
for k in [1,4,12]:
    chosen=next(x for x in runtime_summary if x['n']==selection['selected_n'] and x['chunks']==k)
    full=next(x for x in runtime_summary if x['n']==36 and x['chunks']==k)
    lines.append(f"| {k} | {chosen['median_across9_cells_ms']:.2f} | {full['median_across9_cells_ms']:.2f} | {chosen['median_paired_ratio_to_n36']:.3f}× |")
lines += ['',
    '这里分别取九次独立作业的时间中位数，并汇总作业内的成对时间比，因此表中时间比不必恰好等于两个中位数之比。',
    '当前实现的收益随检索块数量变化，不能把12块的加速推广到1～4块。块内阶段有分组、单独sink和拼接等额外开销；这轮没有优化这些开销。联合层数减少一半也不等于总计算或端到端时间减少一半，因为全部层的MLP、投影和块内attention仍执行。']
lines += ['','## 实现与测量边界','',
    '- 使用之前固定的三任务×8K/32K/128K输入及检索结果；每次选12个512-token块。32题用于选n，剩余68题用于确认，两部分分开报告。没有重新训练或用确认集重新选择n。',
    '- 保留拼接后的RoPE位置，n之后按长度分组独立运行，短尾块不填充。共享sink在块内阶段是单独的块，其他chunk不再读取它；query仍能读取sink和全部历史。',
    '- 通过小型不同长度chunk的完整block-diagonal参考实现，直接检查物理分组计算的K张量，并比较query输出（间接覆盖V）；n36另外与原生Encbank比较。BF16不同计算布局可能有数值差异，记录原生对照分数，不宣称逐位等价。',
    '- 重建时间从H12已就绪到历史第13～36层KV全部就绪，包含分组和KV拼接；不含检索、H12写入、query及decode。每格2次预热、5次计时，先按任务取中位数，再汇总9组中位数与同卡相对n36时间。不同任务/长度的实际尾块长度可能不同。',
    '- 本次调用内保留完整历史上层KV，约48MiB/512-token块；持久H12约4MiB/块。未实现hidden-only decode，也不测跨调用hot buffer命中率。',
    '- 联合计算产生的Hn以及后续KV依赖检索块的有序前缀。未来缓存复用必须带上下文标识，不能按单独chunk ID无条件复用。',
    '- 固定贪心生成上限：NIAH48 token，变量追踪60 token；分数为官方string_match_all参考答案召回率。变量追踪来自Encbank现有生成器，本轮不是完整13任务RULER总分。',
    '- 这里测的是跨chunk attention深度，不能把结果直接当作此前20个线性KV预测头的质量或速度。','',
    '服务器目录：`'+str(R)+'`。所有权重保持在服务器；源码、逐题结果、核验和计时记录在本目录交付。','']
(R/'READOUT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
if '--summarize-only' in sys.argv:
    print(json.dumps(dict(verification=verification,confirmation_tasks={t:x['score'] for t,x in confirmation['tasks'].items()},runtime12=[x for x in runtime_summary if x['chunks']==12]),indent=2))
    sys.exit(0)
files=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json','.txt','.png'] and p.name!='delivery_manifest.json']
for j in jobs:
    out=R/j['run']/'results';files.extend(out.glob('*.json'));files.extend(out.glob('*.jsonl'))
delivery={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
save('delivery_manifest.json',delivery)
with tarfile.open(R/'results_for_local.tar.gz','w:gz') as tar:
    for name in list(delivery)+['delivery_manifest.json']:tar.add(R/name,arcname=name)
print(json.dumps(dict(verification=verification,confirmation_tasks=confirmation['tasks'],runtime=runtime_summary),indent=2))
