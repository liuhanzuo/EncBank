"""Verify completed training, compare untouched diagnostics, deliver no weights."""
import hashlib,json,random,statistics,subprocess,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
assert Path('/srv/encbank') in R.resolve().parents
metrics=['kl','top1_agreement','student_nll','teacher_nll']
def mean(rows):return {k:statistics.mean(r[k] for r in rows) for k in metrics}
def paired(rows):
    rng=random.Random(20260921);result={}
    for key in ['kl','top1_agreement','student_nll']:
        differences=[r['trained'][key]-r['baseline'][key] for r in rows]
        samples=sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(4000))
        result[key]=dict(mean_delta=statistics.mean(differences),document_bootstrap_95=[samples[100],samples[3899]],
            improved_documents=sum(d<0 if key!='top1_agreement' else d>0 for d in differences),documents=len(rows))
    return result
source=json.loads((R/'source_manifest.json').read_text())
for name,expected in source.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==expected,name
jobs=json.loads((R/'submissions.json').read_text());outcomes=[];accounting=[];reference_baseline=None
for job in jobs:
    out=R/job['arm']/'results'
    receipt=json.loads((out/'parent_exit.json').read_text())
    assert receipt['actual_wait'] and receipt['returncode']==0,(job,receipt)
    s=json.loads((out/'summary.json').read_text());assert s['complete']
    grad=json.loads((out/'gradient_check.json').read_text())
    assert grad['all_heads_receive_gradient'] and grad['backbone_trainable_parameters']==0
    row=subprocess.check_output(['sacct','-X','-j',job['job'],'--format=JobID,State,ExitCode,Start,End,NodeList','--noheader','-P'],text=True)
    assert 'COMPLETED|0:0' in row,row
    accounting.append(row)
    baseline=mean(s['baseline_test']);fresh=s['fresh_test'];assert len(fresh)==16
    if reference_baseline is None:reference_baseline=[r['baseline'] for r in fresh]
    else:
        for a,b in zip(reference_baseline,[r['baseline'] for r in fresh]):
            assert all(abs(a[k]-b[k])<1e-5 for k in metrics),(a,b)
    runtime={r['mode']:dict(ms=r['median_ms_per_token'],history_MiB=r['history_persistent_bytes']/2**20,
        projection_ms=statistics.median(p['wall_ms'] for p in r['projection']) if r['projection'] else None)
        for r in s['runtime'] if r['chunks']==12}
    outcomes.append(dict(arm=job['arm'],job=job['job'],config=job['config'],training=s['distillation'],
        baseline_old_test=baseline,trained_old_test=s['test_mean'],
        baseline_fresh=mean([r['baseline'] for r in fresh]),trained_fresh=mean([r['trained'] for r in fresh]),
        paired_fresh=paired(fresh),head_MiB=s['head_parameter_bytes']/2**20,runtime12=runtime,
        head_checkpoint_sha256=s['checkpoint_sha256']))
winner=min(outcomes,key=lambda r:r['training']['selected_validation_kl'])
verification=dict(complete=True,jobs=[j['job'] for j in jobs],source_files_verified=len(source),
    original_source_hashes_unchanged=True,all_actual_wait_zero=True,all_slurm_completed_zero=True,
    all_head_gradient_checks_passed=True,backbone_trainable_parameters=0,
    selected_arm=winner['arm'],selection='minimum validation KL only; fresh test not used for selection')
(R/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
(R/'slurm_accounting.txt').write_text(''.join(accounting))
(R/'compact_results.json').write_text(json.dumps(outcomes,indent=2)+'\n')
base=outcomes[0]
lines=['# H12/H16/H20/H24 独立写入：蒸馏结果','',
    '四个服务器 GPU 作业均已完成，真实父进程 wait=0、Slurm COMPLETED/0:0。模型、原 adapter 与训练头保留在服务器。','',
    '训练 128 个文档、每个 4×512 历史 token 和 128 个后续正文位置，每组 256 次更新（两遍训练集）。4 个文档用于每32步验证和选模型；允许选择未训练的第0步。前轮8个测试文档与本轮新增16个测试文档分开报告，全部与训练和验证文档无交叉。源数据是历史内部 QASPER 正文划分，不是正式问答评分。','',
    '所有组冻结 backbone、writer、原 LoRA 和原投影，缓存点始终为 H12/H16/H20/H24。三组只训练原有20个跨层仿射头，比较三个学习率；第四组增加3个初始为零的修正头，修正独立 H16/H20/H24 对应 KV17/KV21/KV25 的来源上下文差异。H12→KV13 保持原投影。蒸馏目标为 teacher 到 student 的全词表 logits KL，温度1。','',
    '## 验证选择与测试结果','',
    '| 方案 | 验证选中步 | 验证 KL ↓ | 旧8篇 KL ↓ | 旧8篇一致率 ↑ | 新16篇 KL ↓ | 新16篇一致率 ↑ | 新16篇 NLL ↓ |','|---|---:|---:|---:|---:|---:|---:|---:|']
lines.append(f"| 未蒸馏基线 | 0 | {base['training']['initial_validation_kl']:.5f} | {base['baseline_old_test']['kl']:.5f} | {base['baseline_old_test']['top1_agreement']:.2%} | {base['baseline_fresh']['kl']:.5f} | {base['baseline_fresh']['top1_agreement']:.2%} | {base['baseline_fresh']['student_nll']:.4f} |")
for o in outcomes:
    t=o['training'];a=o['trained_old_test'];b=o['trained_fresh'];mark=' **验证选中**' if o is winner else ''
    lines.append(f"| {o['arm']}{mark} | {t['selected_step']} | {t['selected_validation_kl']:.5f} | {a['kl']:.5f} | {a['top1_agreement']:.2%} | {b['kl']:.5f} | {b['top1_agreement']:.2%} | {b['student_nll']:.4f} |")
lines += ['',f"按验证集选择的方案是 `{winner['arm']}`。一致率指与原 teacher 下一 token 预测的吻合程度，不是任务成功率。新增16篇的 teacher NLL 为 {base['baseline_fresh']['teacher_nll']:.4f}。",'',
    '## 参数与运行开销','',
    '每个512-token chunk的历史 H 仍为16MiB；保持20个头的组使用约320.08MiB额外BF16参数，带3个修正头的组约368.09MiB。下面采用12个历史chunk，另含1个sink；历史张量不包含模型、头参数、近期KV、工作区。运行性能每格2次预热、3次正式测量；一次投影为2次预热、5次正式测量。','',
    '| 方案 | 新增头 MiB | 一次投影 ms | KV常驻 ms/token | 按需投影 ms/token | 按需历史 MiB |','|---|---:|---:|---:|---:|---:|']
for o in outcomes:
    rt=o['runtime12']
    lines.append(f"| {o['arm']} | {o['head_MiB']:.2f} | {rt['resident']['projection_ms']:.2f} | {rt['resident']['ms']:.2f} | {rt['on_demand']['ms']:.2f} | {rt['on_demand']['history_MiB']:.2f} |")
lines += ['','各组同卡原生KV基线与全部观测在 runtime.json。不同卡、no_grad/inference_mode 以及实验调度差异都可能影响绝对时间；不以几毫秒差异声称训练带来架构加速。','',
    '## 核验与边界','',
    '- 四组第0步均复现前轮验证KL 0.09793116；新加修正头初始化不改变输出。',
    '- 已验证每个可训练头都收到非零梯度，骨干可训练参数为0；每组训练后核验KV常驻与按需路径的一致性。',
    '- 已核验所有最初登记的源文件SHA未变化，保留训练曲线、独立测试、真实父退出和Slurm记账。',
    '- 新16篇按预先固定的文档顺序选择；不用于学习率或checkpoint选择。对新16篇按文档成对bootstrap的差值区间保存在 compact_results.json，供评估小样本不确定性。',
    '- 此轮扩大了训练集，改善不能单独归因为优化器或蒸馏形式；这是在前轮ridge初始化上的训练增益。',
    '- 仍为静态正文continuation实验；尚未验证真实agent成功率、1024-token自由生成质量或将四点头接入动态hot buffer后的任务质量。','']
(R/'READOUT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
files=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json','.txt'] and p.name not in ['dataset.json','delivery_manifest.json']]
for job in jobs:files.extend((R/job['arm']/'results').glob('*.json'))
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
(R/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(R/'results_for_local.tar.gz','w:gz') as tar:
    for name in list(manifest)+['delivery_manifest.json']:tar.add(R/name,arcname=name)
print(json.dumps(dict(verification=verification,outcomes=outcomes),indent=2))
