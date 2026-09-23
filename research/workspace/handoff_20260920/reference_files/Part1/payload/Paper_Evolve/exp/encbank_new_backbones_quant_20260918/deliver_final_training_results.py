"""Publish complete final-checkpoint scores in the requested results document."""
import datetime,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TARGET=ROOT.parent/'encbank_new_backbones_formal_20260915/delivery/FIVE_BENCHMARK_RESULTS_zh.md'
def main():
    reports=[json.loads((ROOT/f'delivery/{name}_final_generation/five_benchmark_summary.json').read_text(encoding='utf-8')) for name in ('qwen35','qwen38')]
    assert all(r['full_five_benchmark_complete'] and r['errors']==0 and r['cpu_rescored']==5250 and r['judge_records']==1986 for r in reports)
    judge=json.loads((ROOT/'delivery/final_judge_local/monitor.json').read_text(encoding='utf-8'))
    assert judge['parent_exit']['actual_wait'] and judge['parent_exit']['returncode']==0 and judge['complete']['decisions']==11916
    now=datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    section=f'''## 加强训练：最终8,000步（{now}）

两模型的 rank128 / alpha128、固定8,000步版本均已完成全部五项。9B固定j=6，27B固定j=21；每模型7,236条有效唯一答案、1,986条LoCoMo判分。每模型5,250条自动指标在本机CPU执行原评分函数逐条复算，与生成时保存的分数全部一致。八个生成分片均有真实正常退出及Slurm 0:0记录。存储故障前的有效答案保留，恢复不改变科学配置。

| 模型 | Encbank训练版本（H16） | RULER | LongEval | LongBench F1 | BABILong | LoCoMo | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|
'''
    baseline=[('75.53','97.20','46.52','63.62','46.58','65.89'),('99.39','97.20','50.91','73.67','50.70','74.37')]
    order=['ruler','longeval','longbench','babilong','locomo','avg']
    for r,b in zip(reports,baseline):
        section+='| '+r['model']+' | rank32 / alpha32 / 4,000步 | '+' | '.join(b)+' |\n'
        section+='| '+r['model']+' | rank128 / alpha128 / 8,000步 | '+' | '.join(f"{r['scores'][k]:.2f}" for k in order)+' |\n'
    section+='\n这是同时扩大LoRA容量和训练步数的版本对比，不能单独归因为rank或训练时长。9B的RULER与BABILong提高，LongEval持平，LongBench与LoCoMo略降，Avg.从65.89升至66.30。27B的LongEval从97.20升至97.40，LongBench从50.91升至51.08，但LoCoMo从50.70降至49.40，Avg.从74.37降至74.18；增加容量与步数没有带来一致改善。Avg.用未舍入值计算，仅为描述性等权平均。两个模型均报告预定8,000步终点，不根据中途测试分数选择权重。\n\n'
    section+='| 最终版本 | LoCoMo全量（1986题） | 类别1–4（1540题） | 类别5（446题） |\n|---|---:|---:|---:|\n'
    for r in reports:
        section+=f"| {r['model']}，rank128 / 8,000步 | {r['scores']['locomo']:.2f} | {r['locomo_categories']['C1-4']['score']:.2f} | {r['locomo_categories']['C5']['score']:.2f} |\n"
    section+='\nLoCoMo沿用固定gpt-6-astra/low语义评判和类别5本地拒答规则，全量按1986题加权，评分错误为0。\n\n'
    for r,name in zip(reports,('qwen35','qwen38')):
        base=(ROOT/f'delivery/{name}_final_generation').as_posix()
        section+=f"{r['model']}来源：[五项逐单元核验]({base}/five_benchmark_summary.json)、[完整预测]({base}/predictions.jsonl)、[完整LoCoMo判分]({base}/judge_decisions.jsonl)、[四分片正常退出记录]({base}/summary.json)。原评分源码在同目录scoring_sources/。\n\n"
    section+=f"最终评分恢复与完整缓存/尝试备份见[Judge交付记录]({(ROOT/'delivery/final_judge_local/monitor.json').as_posix()})。\n\n"
    text=TARGET.read_text(encoding='utf-8')
    start=text.index('## 加强训练：最终8,000步')
    end=text.index('## 加强训练：中途 LongEval',start)
    text=text[:start]+section+text[end:]
    old='最终8,000步全五项尚无完整结果；截至09月19日06:04，shard3已正常完成1809条，原故障路径与节点检查通过、9B收尾成功后，剩余三个27B分片已恢复运行。完整结果与Judge尚未齐全，不根据中途得分选择权重。'
    assert old in text
    text=text.replace(old,'最终8,000步全五项现已完整交付，见上表；本段保留第4,000步诊断，不根据中途得分选择权重。')
    old='9B固定第8,000步最终权重的五项结果已在上方单列；27B最终版本仍在生成和评分。'
    assert old in text
    text=text.replace(old,'两模型固定第8,000步最终权重的五项结果均已在上方单列。')
    assert '27B最终版本尚未完成' not in text and '仍在生成和评分' not in text
    TARGET.write_text(text,encoding='utf-8')
    receipt=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),document=str(TARGET),complete=True,models=[dict(model=r['model'],scores=r['scores']) for r in reports])
    (ROOT/'delivery/final_training_delivery.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))
if __name__=='__main__':main()
