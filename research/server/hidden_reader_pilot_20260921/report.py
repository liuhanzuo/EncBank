import hashlib,json,statistics,subprocess,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
jobs=json.loads((R/'submissions.json').read_text());summaries=[];accounting=[]
names=dict(exact='逐层 H12–H35',h12='H12 + 逐层拟合头',dual='H12/H24 + 分段拟合头',h24_late='前12层精确KV + H24后段拟合')
for job in jobs:
    out=R/job['arm']/'results'
    receipt=json.loads((out/'parent_exit.json').read_text())
    assert receipt['actual_wait'] and receipt['returncode']==0,(job,receipt)
    summary=json.loads((out/'summary.json').read_text());assert summary['complete']
    row=subprocess.check_output(['sacct','-X','-j',job['job'],'--format=JobID,State,ExitCode,Start,End,NodeList','--noheader','-P'],text=True)
    assert 'COMPLETED|0:0' in row,row
    accounting.append(row);summaries.append(summary)
(R/'slurm_accounting.txt').write_text(''.join(accounting))
lines=['# Qwen3-8B hidden Reader：快速试验结果','',
    '4 组独立单 GPU 作业均完成，真实父进程 wait=0，Slurm COMPLETED / 0:0。模型与原 LoRA 同前轮实验，训练出的投影权重仅保存在服务器。','',
    '这是一轮正文 continuation 的小规模线性可行性试验：32 个文档拟合、4 个文档选择正则强度、8 个文档 held-out 测试。拟合 16,384 个历史位置；使用各层归一化前 K/V 监督的闭式仿射 ridge 拟合，没有做最终 logits 蒸馏，没有训练 query，也没有 agent 任务评分。','',
    '每个测试文档为 4×512 个历史 token，后续正文 128 个输入位置；KL 和 top-1 一致率衡量对原 Encbank teacher 的接近程度，NLL 衡量正文下一 token 的负对数似然，均不是任务成功率。','',
    '| 策略 | 验证选定 ridge | 测试 KL ↓ | 对 teacher 的 top-1 一致率 ↑ | 测试 NLL ↓ | teacher NLL |',
    '|---|---:|---:|---:|---:|---:|']
for s in summaries:
    m=s['test_mean'];r=s['selected_ridge']
    lines.append(f"| {names[s['arm']]} | {r if r is not None else '—'} | {m['kl']:.6f} | {m['top1_agreement']:.2%} | {m['student_nll']:.4f} | {m['teacher_nll']:.4f} |")
lines += ['', '## 相同 checkpoint 的缓存策略比较','',
    'PG19 固定输入；64-token query 后执行 32 个固定输入 decode steps。每格 2 次预热 + 3 次计时。以下为每步中位耗时；历史持久字节包含一个 sink，不包含模型/投影权重、近期 KV 和临时 workspace。每组的原生 KV 基线同卡测量，各组可能位于不同物理 GPU。','',
    '| 策略 | chunks | 原生 KV decode ms/token | 投影KV常驻 ms/token | 按需投影 ms/token | 常驻模式历史 MiB | 按需模式历史 MiB | 一次投影中位 ms |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for s in summaries:
    for n in [1,4,12]:
        cells={x['mode']:x for x in s['runtime'] if x['chunks']==n}
        a,b,c=[cells[k] for k in ['native_KV','resident','on_demand']]
        proj=statistics.median(x['wall_ms'] for x in b['projection'])
        lines.append(f"| {names[s['arm']]} | {n} | {a['median_ms_per_token']:.3f} | {b['median_ms_per_token']:.3f} | {c['median_ms_per_token']:.3f} | {b['history_persistent_bytes']/2**20:.3f} | {c['history_persistent_bytes']/2**20:.3f} | {proj:.3f} |")
lines += ['', '## 核验与限制','',
    '每个作业先以所有 24 层 H 恢复 KV，与原 attention cache 比较，相对 RMS 要求 <1e-5。各策略在第一份测试文档上比较 KV 常驻/按需两种路径，要求 logits KL 绝对值 <1e-4、top-1 一致率 >99%。详细数值在各组 exact_control.json 和 test.json。','',
    '逐层 H 采用固定 teacher 上下文，仅执行原 Norm、KV projection、K norm、RoPE 即可恢复；它保存的 H 比 GQA KV 更大。H12 组仅保存一份 H12；双深度组保存 H12/H24；h24_late 组额外保留前12个上层的精确 KV，不能将其存储成本按仅 H24 计算。','',
    'H24 从同一个 teacher 联合历史上下文生成。该成本未计入投影/逐 token 时间，且不保证改变检索组合后仍可直接复用。所有头与激活不下载。本轮未实现投影吸收的 latent attention kernel；按需模式仍然在每步临时生成本层 memory KV。','',
    'QASPER 源 train/dev 是历史内部文档划分，全部内容仅作正文语言建模；不使用问答答案。训练集已排除全部 dev 文档，验证/测试文档无交叉。它不是正式 QASPER 准确率，也不是对完整模型蒸馏能力的上限估计。','',
    '生成示例仅为两个测试文档各32-token 自由生成诊断；性能测试固定生成输入，未运行真实 agent、动态检索、替换策略或1024-token 长程质量实验。','',
    '各次 CUDA allocated/reserved 峰值保存在 runtime.json，包含模型、当前投影权重、近期 KV 与临时张量；不能与表中的历史持久张量混为一个口径。','']
(R/'RESULTS_zh.md').write_text('\n'.join(lines),encoding='utf-8')
files=['prepare.py','launch.py','experiment.py','observe.py','report.py','README.md','RESULTS_zh.md',
    'dataset_manifest.json','source_manifest.json','submissions.json','queue_before.txt','slurm_accounting.txt']
for job in jobs:
    arm=job['arm']
    files += [arm+'/results/'+n for n in ['environment.json','owner.json','child.json','parent_exit.json',
        'status.json','exact_control.json','test.json','generations.json','runtime.json','summary.json']]
    if (R/arm/'results/validation_selection.json').exists():files += [arm+'/results/validation_selection.json']
manifest={name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in files}
(R/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(R/'results_for_local.tar.gz','w:gz') as tar:
    for name in files+['delivery_manifest.json']:tar.add(R/name,arcname=name)
print(json.dumps([dict(arm=s['arm'],ridge=s['selected_ridge'],metrics=s['test_mean'],runtime=[dict(chunks=r['chunks'],mode=r['mode'],ms=r['median_ms_per_token'],history_MiB=r['history_persistent_bytes']/2**20) for r in s['runtime']]) for s in summaries],indent=2))
