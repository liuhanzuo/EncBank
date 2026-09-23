"""Update only the fully verified 27B quantization rows in the requested delivery file."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=ROOT.parent/'encbank_new_backbones_formal_20260915/delivery/FIVE_BENCHMARK_RESULTS_zh.md'
s=json.loads((ROOT/'delivery/qwen38_quant_five_benchmarks/summary.json').read_text())
assert s['local_aggregation_verified'] and s['predictions']==14472 and s['judge_records']==3972 and s['errors']==0
text=p.read_text(encoding='utf-8');before=text
order=['ruler','longeval','longbench','babilong','locomo','avg']
for bit in [8,4]:
    prefix='| Qwen3.8-27B | Encbank-H%d |'%bit
    matches=[line for line in text.splitlines() if line.startswith(prefix)]
    assert len(matches)==1,'Only replace unique main quantization row before adding LoCoMo detail'
    x=s['scores']['cache_h%d'%bit]
    text=text.replace(matches[0],prefix+' '+' | '.join('%.2f'%x[k] for k in order)+' |',1)
text=text.replace('## 缓存量化补测：全部五项 benchmark（2026-09-18）','## 缓存量化补测：全部五项 benchmark（2026-09-19）',1)
text=text.replace('9B 的 H8/H4 每组均完成7,236题','9B与27B的H8/H4每组均完成7,236题',1)
text=text.replace('共14,472条预测，0 OOM、0未评分。','每模型共14,472条预测，两模型合计28,944条，0 OOM、0未评分。',1)
text=text.replace('27B 的 H8/H4 全五项仍在运行；只在完整分母与评分齐全后填入对应成绩，不用已完成子集外推正式分数。','两模型H8/H4的全五项已完成生成、LoCoMo Judge与CPU逐题复算，并在本地独立核对完整分母和各单元宏平均。',1)
anchor='| Qwen3.5-9B | Encbank-H4 | 46.78 | 56.62 | 12.78 |'
assert text.count(anchor)==1
rows=[]
for bit in [8,4]:
    d=s['locomo_breakdown']['cache_h%d'%bit]
    rows.append('| Qwen3.8-27B | Encbank-H%d | %.2f | %.2f | %.2f |'%(bit,d['all']['score'],d['C1-4']['score'],d['C5']['score']))
text=text.replace(anchor,anchor+'\n'+'\n'.join(rows),1)
text=text.replace('两组各1,986题，共3,972条判分均已完成，评分错误为0。全量正确数分别为923和929，类别1–4正确数分别为866和872，类别5均为57；分数均为0–100。',
    '每模型两组各1,986题、3,972条判分，两模型共7,944条判分均已完成，评分错误为0。9B全量正确数分别为923和929，类别1–4正确数分别为866和872，类别5均为57；27B逐题判断及分类分母见下方交付记录。分数均为0–100。',1)
anchor='## 加强训练：中途 LongEval 诊断（2026-09-19 02:58）'
assert text.count(anchor)==1
sources='27B来源：[全五项汇总与核验](F:/Paper_Evolve/exp/encbank_new_backbones_quant_20260918/delivery/qwen38_quant_five_benchmarks/summary.json)、[原始预测及评分交付包](F:/Paper_Evolve/exp/encbank_new_backbones_quant_20260918/delivery/qwen38_quant_five_benchmarks/saved_outputs.tar.gz)、[逐题LoCoMo判分](F:/Paper_Evolve/exp/encbank_new_backbones_quant_20260918/delivery/qwen38_quant_five_benchmarks/judge_decisions.jsonl)、[任务正常退出证据](F:/Paper_Evolve/exp/encbank_new_backbones_quant_20260918/delivery/qwen38_quant_five_benchmarks/exit_evidence.json)。\n\n'
text=text.replace(anchor,sources+anchor,1)
backup=ROOT/'delivery/qwen38_quant_five_benchmarks/FIVE_before_update.md'
assert not backup.exists(),'One-shot table update already attempted; inspect current delivery'
backup.write_text(before,encoding='utf-8');p.write_text(text,encoding='utf-8')
print(p)
