from pathlib import Path
import json
from io_utils import save
ROOT=Path(__file__).resolve().parent
def main():
    rows=json.loads((ROOT/'collected/complete.json').read_text())['results']
    environment=json.loads((ROOT/'collected/environment.json').read_text())
    plan=json.loads((ROOT/'plan.json').read_text())
    lines=['# Qwen3.8-27B Encbank k48 单卡容量','',
        '同一模型只加载一份，j21、原final4000 adapter；每会话独立GPU hidden bank，读取48个512-token块，变长query8192/1024/2048；无offload。',
        f"驱动报告 {environment['gpu']}，物理显存 {environment['total_memory_bytes']/2**30:.2f} GiB；本实验统一限制allocator为 {plan['device_cap_gib']} GiB。OOM表示当前执行器在这一额度内失败，不代表物理显存全部耗尽。",
        '长KV压力组补分配32704个零值attention缓存位置，再执行63个真实decode forward；它测容量/带宽压力，不是自然生成32768 tokens或质量测评。源H库由真实编码的512-token块复制到完整独立分配容量，不测完整Write耗时。','',
        'prefill逐会话执行、decode异长合批；峰值包含KV合并时的临时副本。每条件单次容量试验，63步强制token decode，吞吐包含该点的首步开销，不能视为稳定生产吞吐。','',
        '| 每路source | 额外KV槽位 | 并行数 | 状态 | allocated GiB | reserved GiB | NVML采样 GiB | 总decode tok/s |',
        '|---:|---:|---:|---|---:|---:|---:|---:|']
    summaries=[]
    for source in [32768,131072]:
        for past in [0,32704]:
            good=[]
            for r in rows:
                if (r['source_tokens'],r['prior_decode_attention_slots'])!=(source,past):continue
                a=f"{r['peak_allocated_bytes']/2**30:.2f}" if 'peak_allocated_bytes' in r else '-'
                b=f"{r['peak_reserved_bytes']/2**30:.2f}" if 'peak_reserved_bytes' in r else '-'
                t=f"{r['aggregate_decode_tps']:.2f}" if 'aggregate_decode_tps' in r else '-'
                nv=f"{r['nvml_sampled_peak_bytes']/2**30:.2f}" if 'nvml_sampled_peak_bytes' in r else '-'
                lines.append(f"| {source} | {past} | {r['batch']} | {r['status']} | {a} | {b} | {nv} | {t} |")
                if r['status']=='ok':good.append(r['batch'])
            summaries.append(dict(source_tokens=source,prior_decode_attention_slots=past,largest_tested_successful_batch=max(good,default=0)))
    save(ROOT/'summary.json',dict(conditions=summaries,scope='Measured capacity stress bound; not a natural-trajectory production concurrency maximum',environment=environment,allocator_cap_gib=plan['device_cap_gib'],rows=rows))
    lines+=['','已验证的并行下限：','', '| 每路source | 额外KV槽位 | 最大实测通过档 |', '|---:|---:|---:|']
    for s in summaries:lines.append(f"| {s['source_tokens']} | {s['prior_decode_attention_slots']} | {s['largest_tested_successful_batch']} |")
    lines+=['','如果最大测试档通过，只能说至少支持该档；不能声称已经找到最大并发。skipped_after_oom 是较小档OOM后未测，并非又一次实测OOM。实际Terminal-Bench还受生成长度、工具等待、CPU容器内存和调度策略限制。',
        '原始结果中的query_lengths沿用了加上额外KV槽位后的position计数；原始query长度为8192/1024/2048循环，不是把长KV压力组的零值槽位当成真实query。NVML为100ms采样。']
    (ROOT/'RESULTS_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(summaries))
if __name__=='__main__':main()
