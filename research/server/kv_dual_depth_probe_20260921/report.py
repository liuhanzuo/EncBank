"""Summarize completed measurements; no inference, no source-result mutation."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

root = Path(__file__).resolve().parent
out = root/'results'
summary = json.loads((out/'summary.json').read_text())
receipt = json.loads((out/'parent_exit.json').read_text())
validation = json.loads((out/'validation.json').read_text())
assert summary['complete'] and receipt['actual_wait'] and receipt['returncode'] == 0
assert validation['passed'] and len(validation['checks']) == 90
accounting = (out/'slurm_accounting.txt').read_text()
assert 'COMPLETED|0:0' in accounting
env = summary['environment']
cells = {(r['graph'],r['chunks'],r['method']):r for r in summary['cells']}
raw = [json.loads(line) for line in (out/'measurements.jsonl').read_text().splitlines()]
assert len(raw) == 1260
checks = validation['checks']
lines = ['# H12/H24 双深度 KV 重建实测', '',
    f"作业 {env['job']}；{env['gpu']} 单卡；Qwen3-8B / 36 层 / 512-token chunk；BF16 backbone + 原 FP32 LoRA。父进程实际 wait=0，Slurm COMPLETED / 0:0。", '',
    '本次实现没有达到耗时减半。双线程 + 双 stream 在独立 chunk batch 中缩短约 4.6%–15.1%，联合重建中缩短约 1.9%–11.5%。这是该模型和当前 PyTorch/LoRA 执行路径的实测，不是所有实现的理论上限。', '',
    'H12 基线串行执行第 13–36 层。双深度方法预存 H24，一路 H12 → 第 13–24 层，另一路 H24 → 第 25–36 层。两路仍在同一张 GPU 上执行，最后合并各层 KV 引用。', '',
    '3 份保存的 PG19 输入，每格每份输入 2 次预热 + 7 次正式计时，方法顺序随机交错；共 1,260 条正式观测。表中为 21 次 CUDA 同步 wall time 的中位数，单位 ms。', '']

def value(graph,count,method):return cells[(graph,count,method)]['wall_ms']['median']

derived = []
for graph,title in [('independent_batch','独立 chunk 批量重建'),('joint_pack','固定检索组合的联合重建')]:
    lines += ['## '+title, '',
        '| chunks | H12 基线 | 双深度串行 | 双 stream | 双线程 + 双 stream | 双线程加速比 | 独立两半理想参考 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for count in [1,2,4,8,12]:
        base,serial,streams,threads,half_a,half_b = [value(graph,count,m) for m in
            ['h12_full','dual_serial','dual_streams','dual_threads','half_13_24','half_25_36']]
        ideal = max(half_a,half_b)
        lines.append(f'| {count} | {base:.3f} | {serial:.3f} | {streams:.3f} | {threads:.3f} | {base/threads:.3f}× | {ideal:.3f} |')
        derived.append(dict(graph=graph,chunks=count,h12_ms=base,dual_serial_ms=serial,
            dual_streams_ms=streams,dual_threads_ms=threads,threads_speedup=base/threads,
            streams_speedup=base/streams,threads_time_reduction_fraction=1-threads/base,
            half_13_24_ms=half_a,half_25_36_ms=half_b,ideal_reference_ms=ideal))
    lines += ['']
    if graph == 'independent_batch':
        lines += ['每个 chunk 为一个独立 batch row，chunk 内因果注意力；它适合测试独立重建块的计算代价，但与当前跨 chunk 联合注意力不同。','']
    else:
        lines += ['含 1 个 sink token，所选 chunks 联合因果注意力。这里的 H24 事先按完全相同的组合、顺序、位置和 batch shape 生成。该结果不能证明改变检索组合后，之前存下来的 H24 仍可直接复用。','']

max_diff = max(x['max_abs'] for x in checks)
max_rel = max(x['relative_rms'] for x in checks)
unequal = sum(x['unequal_elements'] for x in checks)
lines += ['## 一致性和解释范围', '',
    f'90 组完整上层 KV 对照通过。最大绝对误差 {max_diff:.8g}；最大相对 RMS {max_rel:.8g}；累计不完全相等元素 {unequal}。这是相同 attention graph 内的 KV 检查，不是 agent 任务质量实验。', '',
    '“独立两半理想参考”取两个 12 层段各自单独运行的较大值；它只是没有资源争用与额外开销时的参考，不是单卡可以保证达到的延迟。双 stream 事件包络有重叠，也不能直接证明两个 GPU kernel 真正同时执行。', '',
    '计时覆盖 GPU resident hidden 到 24 层 KV 完成及引用合并，不含检索、H 的初次编码/传输、query、LM head、hot buffer 组装或 agent 环境交互。H24 准备成本单独保存在 preparation.json。双深度并行没有减少总层计算量，同一卡的算力、显存带宽和 host 提交开销仍会限制加速。', '',
    'BF16、hidden size=4096 时，每个 512-token chunk 的 H12 为 4 MiB，额外 H24 为 4 MiB，双 hidden 合计 8 MiB；24 个上层的 KV 仍为 48 MiB/chunk。', '',
    '用户 hot-buffer 设计约束：命中以 hidden memory 检索出的 chunk 是否已有可用 hot KV 为依据；当前生成 chunk 的 KV 也纳入 hot buffer，生成期间追加，完成后可供后续检索命中。此次未测真实轨迹命中率，未实施替换策略或插入逻辑。表中 chunks 是实际需要重建的数量，不是生成长度或检索命中率。', '',
    '不下载权重、checkpoint 或 adapter；既有 benchmark 服务及之前重建实验不做修改。', '']
(out/'RESULTS_zh.md').write_text('\n'.join(lines),encoding='utf-8')
(out/'comparison.json').write_text(json.dumps(derived,indent=2)+'\n')
with (out/'timings.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(derived[0]));writer.writeheader();writer.writerows(derived)
files = ['benchmark.py','launch.py','prepare_submit.py','report.py','README.md','source_manifest.json',
    'submission.json','submission_intent.json','job.sh']
files += ['scheduling_before.txt','scheduling_after.txt']
files += ['results/'+name for name in ['summary.json','environment.json','validation.json','preparation.json',
    'measurements.jsonl','owner.json','child.json','parent_exit.json','status.json','RESULTS_zh.md',
    'comparison.json','timings.csv','slurm_accounting.txt']]
manifest={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
(root/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(root/'results_for_local.tar.gz','w:gz') as tar:
    for name in files+['delivery_manifest.json']:tar.add(root/name,arcname=name)
print(json.dumps(dict(job=env['job'],formal_observations=len(raw),max_abs=max_diff,
    max_relative_rms=max_rel,unequal_elements=unequal,comparisons=derived),indent=2))
