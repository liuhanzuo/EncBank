"""Create readable reports and a hash-verified, weights-free delivery archive."""
import hashlib,json,statistics,subprocess,sys,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
LONG=R.parent/'hidden_reader_longkd_20260921'
def save(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n')
def package(root):
    files=[p for p in root.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json','.txt','.png']
        and p.name not in ['dataset.json','delivery_manifest.json']]
    jobs=json.loads((root/'submissions.json').read_text())
    for j in jobs:
        out=root/j.get('cell',j.get('arm'))/'results';files.extend(out.glob('*.json'));files.extend(out.glob('*.jsonl'))
        receipt=json.loads((out/'parent_exit.json').read_text());assert receipt['actual_wait']
        accounting=subprocess.check_output(['sacct','-X','-j',j['job'],'--format=JobID,State,ExitCode','--noheader','-P'],text=True)
        if receipt['returncode']==0:assert 'COMPLETED|0:0' in accounting,accounting
        else:
            assert receipt['returncode']==1 and 'FAILED|1:0' in accounting,accounting
            assert (out/'predictions.jsonl').stat().st_size==0
            recovery=root/'numerical_control'/j['cell']/'results'
            assert json.loads((recovery/'parent_exit.json').read_text())['returncode']==0
    control=root/'numerical_control'
    if control.exists():
        files.extend(p for p in control.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json','.txt'])
        for j in json.loads((control/'submissions.json').read_text()):
            out=control/j['cell']/'results';files.extend(out.glob('*.json'));files.extend(out.glob('*.jsonl'))
            rec=json.loads((out/'parent_exit.json').read_text());assert rec['actual_wait'] and rec['returncode']==0
            accounting=subprocess.check_output(['sacct','-X','-j',j['job'],'--format=JobID,State,ExitCode','--noheader','-P'],text=True)
            assert 'COMPLETED|0:0' in accounting,accounting
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    save(root/'delivery_manifest.json',manifest)
    with tarfile.open(root/'results_for_local.tar.gz','w:gz') as tar:
        for name in list(manifest)+['delivery_manifest.json']:tar.add(root/name,arcname=name)
    print(json.dumps(dict(root=str(root),files=len(manifest),archive_bytes=(root/'results_for_local.tar.gz').stat().st_size)))
training=json.loads((LONG/'compact_results.json').read_text())['records']
lines=['# 四点 hidden 缓存：延长蒸馏训练','',
    '两组均从先前256步模型继续1792次更新，累计2048步，全部正常结束。骨干、原COMem LoRA、writer和原层投影冻结，只训练20个跨层KV仿射头；缓存H12/H16/H20/H24，不缓存H24之后的hidden。模型权重仅保留在服务器。','',
    '训练集仍为128个QASPER正文文档，每次4×512历史token和128个后续位置。4个文档验证集；旧8篇与前轮新增16篇诊断集分开报告。本轮没有增加文档量。学习率做余弦衰减至峰值的10%，优化器重新初始化，未恢复之前没有保存的优化器状态。每128次额外更新验证，起始256步也可被选中。','',
    '| 方案 | 累计训练步数 | 验证选中步 | 验证KL↓ | 旧8篇KL↓ | 另16篇KL↓ | 另16篇预测一致率↑ |',
    '|---|---:|---:|---:|---:|---:|---:|']
for x in training:
    t=x['training'];total=t.get('total_updates',256);sel=t.get('selected_total_steps',256)
    lines.append(f"| {x['arm']} | {total} | {sel} | {t['selected_validation_kl']:.5f} | {x['old_test']['kl']:.5f} | {x['fresh_test']['kl']:.5f} | {x['fresh_test']['top1_agreement']:.2%} |")
lines += ['',
    '按验证KL选中小学习率3e-6的累计640步模型。大峰值学习率1e-5没有超过起始256步。旧测试KL改善约2.75%，另16篇KL改善约8.64%；这些是正文续写分布诊断，不代表问答正确率。验证集仅4个文档，选中点的稳定性仍有限。','',
    '完整2048步端点另外保存，RULER质量评测对比原COMem、256步、验证选中640步、同一训练分支2048步端点。没有使用RULER分数来选择训练分支或检查点。','',
    '训练每组约8.6分钟，含加载、诊断与性能测量的Slurm总时长约11.5分钟。训练源文件SHA已逐一复核，真实父进程wait和Slurm均为成功退出。','',
    '服务器根目录：`'+str(LONG)+'`。训练曲线见各组`results/distill_validation.json`和`results/distill_trace.json`；汇总见`compact_results.json`。','']
(LONG/'READOUT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
package(LONG)
if '--training-only' in sys.argv:sys.exit(0)
s=json.loads((R/'scores.json').read_text())
lines=['# 四点 hidden→KV：RULER质量对照','',
    '这批测试显示当前四点hidden投影方案存在明显任务质量损失；正文续写KL改善没有转化为多键检索和变量追踪能力的恢复。增加到2048步没有解决这一问题，当前结果不支持将它作为原COMem的无损替代。','',
    'Qwen3-8B + 原COMem LoRA，900道固定题目，四组主对照共3600次独立贪心生成，另外900次真实KV分开处理对照，共4500次有效生成。各组共享原始题目、分块、检索结果、提示词、EOS、生成长度上限与评分函数。原COMem是本次同环境重新跑出的基线。','',
    '训练分支和640步检查点按独立验证集KL提前选定；2048步是同一分支的预先指定末尾端点。RULER题目未用于蒸馏或选检查点。','',
    '## 得分','',
    '得分为参考答案字符串召回率（%）；每格100题。单目标和多键任务每题一个目标，得分相当于答对比例；变量追踪每题5个变量，允许部分得分。多键任务仍是单答案检索，不是多答案NIAH。','',
    '| 任务／原始上下文长度 | 原COMem | 真KV分开处理 | 256步 | 验证选中640步 | 2048步 | 640步相对COMem |',
    '|---|---:|---:|---:|---:|---:|---:|']
labels={'single':'单目标NIAH','multikey':'多键干扰NIAH','vt':'变量追踪（COMem版）'}
for cell,x in s['cells'].items():
    task=next(t for t in labels if cell.startswith(t));length=cell[len(task):].upper();v=x['score']
    lines.append(f"| {labels[task]} / {length} | {v['comem']:.2f} | {v['comem_split']:.2f} | {v['kd256']:.2f} | {v['kd_selected']:.2f} | {v['kd2048']:.2f} | {x['comparison']['kd_selected']['delta_vs_comem_pp']:+.2f} pp |")
lines += ['','## 成对统计','',
    '| 范围 | 原COMem | 真KV分开处理 | 256步 | 640步 | 2048步 | 640步差值的95%区间 |',
    '|---|---:|---:|---:|---:|---:|---|']
for key,label in [('official_niah_only','NIAH两任务×三长度，600题'),('comem_ruler_style_vt','COMem变量追踪×三长度，300题')]:
    g=s['groups'][key];v=g['score'];ci=g['comparison']['kd_selected']['paired_95ci_pp']
    lines.append(f"| {label} | {v['comem']:.2f} | {v['comem_split']:.2f} | {v['kd256']:.2f} | {v['kd_selected']:.2f} | {v['kd2048']:.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] pp |")
for key,label in [('official_niah_only','NIAH'),('comem_ruler_style_vt','变量追踪')]:
    g=s['groups'][key];c=g['comparison']['kd_selected']
    lines += ['',f"{label}：640步相对原COMem平均{c['delta_vs_comem_pp']:+.2f}个百分点；{c['comem_full_correct_to_student_not_full']}题由完全正确变为非完全正确，{c['student_full_correct_from_comem_not_full']}题反向改善。"]
    ci=c['paired_95ci_vs_split_pp'];num=g['comparison']['comem_split']
    lines += [f"真实KV分开处理对照相对原COMem为{num['delta_vs_comem_pp']:+.2f}个百分点；640步相对该对照为{c['delta_vs_comem_split_pp']:+.2f}个百分点，95%区间[{ci[0]:+.2f}, {ci[1]:+.2f}]。"]
lines += ['',
    '置信区间采用按任务和长度分层的10,000次成对题目bootstrap，未做多重比较校正。平均值相近或区间跨零，都不能证明无质量损失；这里只描述这批固定样本。','',
    '## 失败样例核查','',
    '640步模型的多键任务中，89题由原COMem正确变为错误；这89题的目标答案全部仍在相同的检索文本中。88题首行给出了错误数字，其中84题的错误数字可以在检索文本的其他位置找到。变量追踪中，288题由原先完全正确变为非完全正确，所需变量名全部仍在检索文本中；其中80题首行把问题里的数值重复至少3次。这些是错误模式统计，不能单独证明具体内部机制。详细样例见failure_analysis.json。','',
    '全部4500次生成都达到固定长度上限，没有提前遇到指定EOS。结果只衡量相同48/60-token预算内的答案召回；没有评估更长输出预算能否让学生在后续文本中纠正答案。','',
    '## 对后续实验的含义','',
    '当前训练是4块历史上的正文续写，测试则使用12块检索历史并要求精确选择数字或追踪变量。训练与测试的任务、历史块数均不同；本轮结果不能区分数据覆盖不足与当前投影结构的能力限制。更有价值的下一步是做任务型蒸馏与分层精确KV替换消融，分别检验训练目标和跨层近似的影响。未在本轮根据RULER分数调整模型或继续训练。','',
    '## 配置与范围','',
    '- 原始上下文8K、32K、128K；每组均按512 token分块，iterative BM25选择12块，每轮4块，按文档顺序拼接。模型上层实际读取约6144个历史token加sink和问题，不是128K稠密attention。',
    '- 对选择出的chunk独立写到H12/H16/H20/H24，KV13/17/21/25用相应原投影，其余20层用蒸馏仿射头。H24负责后续直到KV36。生成token的KV由正常逐层decode产生。',
    '- 本轮为质量测试，投影后的历史KV在生成期间常驻。仅对检索命中的chunk进行惰性写入，与提前独立写入的语义相同；结果不能作为整段历史写入耗时、显存总容量或hot buffer命中率的实测。',
    '- 三个学生使用同一架构。增加训练步数不改变每chunk的hidden存储量或投影头参数量。',
    '- 单目标与多键NIAH使用已有固定输入，来自NVIDIA官方RULER生成器；变量追踪来自现有COMem内置RULER风格生成器，单独标明。此处不是完整13任务RULER总分。',
    '- greedy生成：NIAH最多48 token，变量追踪最多60 token；第一步抑制EOS，之后遇151645停止，与原固定测试协议一致。未改提示词或启用新的chat模板。',
    '- 评分直接调用固定版本官方string_match_all，并与逐题评分交叉核对。所有题目与标签ID一一对应，推理进程不读取答案标签。',
    '- 初始9组中4组在首题检查因BF16联合/分开处理差异而停止，没有产生完整答题记录。这4次失败原样保留；追加9组作业，其中4组重新执行主对照并补真KV对照，另5组只补真KV对照，没有替换已有效完成的结果。',
    '- 真KV对照把原生COMem联合prefill的历史KV原样取出，再单独处理问题，使用与学生相同的query/cache调用路径。历史KV逐位不变且缓存长度正确已核验；BF16联合/分开计算本身不保证logits或argmax逐位相同，因此另外测了全部900题的输出，而不将数值差异视为零。',
    '- 原先严格的首题KL<0.001/argmax一致阈值改为记录数值差异并核验KL<0.01、max_abs<1、历史KV逐位不变和缓存长度。该阈值调整不是质量通过标准；质量影响由全部题目的真实KV对照得分单独报告。控制误差见score_checks.json；运行前登记的两批源文件和输入SHA在运行后全部复核。',
    '- 尚未覆盖完整RULER的其他任务、真实agent成功率或超过一个512-token生成chunk后的动态检索。','',
    'RULER来源：[NVIDIA/RULER](https://github.com/NVIDIA/RULER)。详细源文件路径、版本与SHA见input_inventory.json、protocol.json和source_manifest.json。逐题四组输出见scored_items.json，成对区间与失败数量见scores.json。','',
    '服务器根目录：`'+str(R)+'`。所有模型、adapter和训练头保持在服务器；本地交付仅包含代码、结果和证据。','']
(R/'READOUT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
package(R)
