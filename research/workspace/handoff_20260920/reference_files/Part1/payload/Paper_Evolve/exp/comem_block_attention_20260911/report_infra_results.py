"""Audit actual completed cells and produce a compact, readable result table."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/infra_5090_28g'
GIB = 2**30
MIB = 2**20


def main():
    state = json.loads((OUT / 'status.json').read_text(encoding='utf-8'))
    plan = json.loads((OUT / 'plan.json').read_text(encoding='utf-8'))
    assert state['status'] == 'complete'
    assert len(plan) == 24 and len({j['id'] for j in plan}) == 24
    assert set(state['jobs']) == {j['id'] for j in plan}
    identities, input_hashes, rows = {}, {}, []
    native_hashes = set()
    for job in plan:
        receipt = state['jobs'][job['id']]
        assert receipt['status'] == 'complete', job['id']
        result, monitor = receipt['result'], receipt['monitor']
        assert monitor['status'] == 'complete' and monitor['exit_code'] == 0
        assert monitor['samples'] > 0 and not monitor['interference']
        assert result['status'] == 'complete' and result['timing_eligible'] is True
        assert result['hardware']['gpu'] == 'NVIDIA GeForce RTX 5090'
        admission = result['hardware']['admission']
        assert 0 <= admission['initial_used_gib'] < 5
        assert 0 <= admission['recheck_used_gib'] < 5
        assert not admission['other_python_compute_processes']
        assert result['hardware']['gpu_incremental_budget_bytes'] == 28 * GIB
        assert result['hardware']['torch_allocator_cap_bytes'] == 27.5 * GIB
        assert 0 < monitor['peak_incremental_gpu_bytes'] <= 28 * GIB
        assert len(result['requests']) == 1
        request = result['requests'][0]
        assert len(request['generated_ids']) == request['generated_tokens'] == 32
        assert request['decode_steps'] == 31
        summary = result['summary']
        for key in ('ttft_s', 'decode_wall_s', 'query_e2e_s', 'decode_tokens_per_s'):
            assert math.isfinite(summary[key]) and summary[key] > 0
        assert math.isclose(summary['decode_tokens_per_s'], 31 / summary['decode_wall_s'])
        assert math.isclose(summary['query_e2e_s'], summary['ttft_s'] + summary['decode_wall_s'])
        identity = result['identity']
        assert identity['document_tokens'] == job['document_tokens']
        assert identity['prompt_tokens'] == 64 and identity['generation_tokens'] == 32
        # Only native/FULL invoke this backend. Sparse arms deliberately log
        # None rather than claiming that they executed the native wrapper.
        if job['arm'] in ('NATIVE', 'FULL'):
            assert identity['native_reader_sha256']
            native_hashes.add(identity['native_reader_sha256'])
        else:
            assert identity['native_reader_sha256'] is None
        shared = {k: v for k, v in identity.items()
                  if k.endswith('sha256') and k != 'native_reader_sha256'}
        if identities:
            assert shared == identities, 'Methods use different code/model/adapter sources'
        identities = shared
        inputs = result['input']['token_sha256']
        assert input_hashes.setdefault(job['document_tokens'], inputs) == inputs
        rows.append((job, result, monitor))
    assert len(native_hashes) == 1
    parity = json.loads((OUT / 'CACHE_PARITY.json').read_text(encoding='utf-8'))
    assert len(parity['diagnostics']) == 9
    assert all(x['status'] == 'passed_token_diagnostic' for x in parity['diagnostics'])
    checks = dict(checked_utc=datetime.now(timezone.utc).isoformat(), cells=24,
                  complete=24, oom=0, incremental_cap_violations=0,
                  generated_tokens=24*32, decode_steps=24*31,
                  same_source_hashes=True, same_input_within_each_length=True,
                  cache_token_route_parity_pairs=9,
                  maximum_sampled_increment_gib=max(m['peak_incremental_gpu_bytes']/GIB for _, _, m in rows),
                  scope='One synthetic fixed-length engineering measurement per cell; no accuracy or instantaneous driver-memory guarantee')
    (OUT / 'INFRA_CHECKS.json').write_text(json.dumps(checks, indent=2)+'\n', encoding='utf-8')
    names = {'NATIVE': '原生 CoMem + LoRA', 'FULL': 'Full recompute', 'D0': 'D0 同图 wrapper 对照',
             'A': 'A 全块', 'B': 'B 剪枝 50%', 'D1': 'D1 原注意力 + 剪枝'}
    text = ['# CoMem + Sparse Attention：5090 实测表', '',
            '**24/24 配置完成，无 OOM，无采样显存增量超限。** 4K、16K、32K均为实际文档读取量；另有64-token问题和固定32-token输出。每项仅一次合成输入工程测量，使用同一强LoRA，不能代替训练后质量评测。', '',
            '这轮三个长度下，原生CoMem端到端耗时均最低。稀疏方法减少部分活跃张量，但分配器保留量、设备驻留和读取开销仍需优化；不能据解析KV字节宣称总体优势。', '',
            '显存列是包含权重的设备使用量相对GPU准入基线的采样峰值，既有桌面占用不计入增量。分配器硬限为27.5GiB，另留0.5GiB并监控28GiB增量预算；采样不是驱动瞬时硬上限。WDDM进程显存不可用值保留N/A。', '',
            'TTFT包含缓存文件加载、H2D、路由、prefill和首token；decode TPS按31次后续解码计算。E2E为完整一次请求，模型加载和一次文档/热缓存写入单列。Full recompute持久化原token，在线构建全层KV并正常cached decode；它沿用同一upper-layer LoRA。', '']
    for length in (32768, 16384, 4096):
        text += [f'## 实际文档 {length//1024}K', '',
                 '| 方法 / 缓存 | 持久缓存 MiB | 显存增量峰值 GiB | TTFT s | Decode tok/s | 请求 E2E s |',
                 '|---|---:|---:|---:|---:|---:|']
        for job, result, monitor in rows:
            if job['document_tokens'] != length:
                continue
            mode = ('原token' if job['arm'] == 'FULL' else
                    '块热缓存' if job['cache_mode'] == 'block_hot' else 'h_j 冷缓存')
            store = result['store']
            size = (store['cold_file_bytes'] + store['hot_file_bytes']) / MIB
            size_text = f'{size:.3f}' if size < 1 else f'{size:.2f}'
            m = result['summary']
            text.append(f"| {names[job['arm']]} / {mode} | {size_text} | {monitor['peak_incremental_gpu_bytes']/GIB:.2f} | {m['ttft_s']:.3f} | {m['decode_tokens_per_s']:.2f} | {m['query_e2e_s']:.3f} |")
        text.append('')
    detail = (OUT / 'INFRA_REPORT.md').as_posix()
    text += [f'[完整 allocated/reserved、一次写入、热缓存构建和传输明细](<{detail}>)。', '',
             '9组冷/热及D0/native对照的生成token和选块完全一致；这只是所测合成输入上的缓存语义检查。CoMem exact-prefix基线仍待补。128K+仍待共同长窗口配置与质量验证；32K没有替代它。', '',
             '3090训练队列继续等待空卡。B300 Slurm单卡脚本已准备，但SSH停在公钥认证，尚未提交。']
    (OUT / 'FINAL_REPORT.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    # Compact manuscript fragment: use the longest measured setting in the
    # main table and retain all lengths/allocator diagnostics in the report.
    labels = {'NATIVE': 'CoMem + LoRA', 'FULL': 'Full recompute', 'D0': 'D0 control',
              'A': 'A: block-all', 'B': 'B: block-pruned', 'D1': 'D1: dense-pruned'}
    tex = [r'% Synthetic engineering precheck only; booktabs required.',
           r'\begin{table}[t]', r'\centering\small', r'\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}{lrrrrr}', r'\toprule',
           r'Method / cache & \shortstack{Store\\MiB} & \shortstack{$\Delta$VRAM\\GiB} & \shortstack{TTFT\\s} & \shortstack{Decode\\tok/s} & \shortstack{E2E\\s} \\',
           r'\midrule']
    for job, result, monitor in rows:
        if job['document_tokens'] != 32768:
            continue
        label = labels[job['arm']]
        if job['arm'] in ('A', 'B'):
            label += ' / hot' if job['cache_mode'] == 'block_hot' else ' / cold'
        store = result['store']
        size = (store['cold_file_bytes'] + store['hot_file_bytes']) / MIB
        size_text = f'{size:.3f}' if size < 1 else f'{size:.1f}'
        m = result['summary']
        tex.append(f"{label} & {size_text} & {monitor['peak_incremental_gpu_bytes']/GIB:.2f} & {m['ttft_s']:.2f} & {m['decode_tokens_per_s']:.2f} & {m['query_e2e_s']:.2f} " + r'\\')
    tex += [r'\bottomrule', r'\end{tabular}',
            r'\caption{Synthetic 32K-document engineering precheck on one RTX 5090, with a common strong upper-layer LoRA, 64-token query and 32-token output. Each configuration was measured once. Store includes cold and extra hot files; full recompute stores raw tokens. $\Delta$VRAM is the sampled device-memory increase from the pre-CUDA baseline, including weights, not an instantaneous upper bound. TTFT includes loading and transfers; one-time writes and allocator peaks are reported separately.}',
            r'\label{tab:sparse-infra-32k-precheck}', r'\end{table}']
    (OUT / 'INFRA_MAIN_TABLE.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
