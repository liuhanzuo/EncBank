"""Join verified A and existing B/C into one reviewer-facing result file."""
import datetime,json,tomllib
from pathlib import Path
import config
root=config.ROOT
assert json.loads((root/'delivery_verification.json').read_text())['complete']
a=(root/'RESULTS_zh.md').read_text(encoding='utf-8')
b=(root/'existing_analysis/RESULTS_zh.md').read_text(encoding='utf-8')
intro='''# Reviewer 补测回传结果

更新：2026-09-19。必做A已完成600条正式答案；必做B/C由已有2400请求、40更新日志整理完成。仅本轮A新增LLM运行，未新增训练。LoCoMo可选200对语义重评分未启动，不能据本报告声称其语义质量优势已验证。

A统一采用Qwen3-8B原生BF16、同一RTX5090、共享rank32/alpha32/final4000 adapter、BGE-large-en-v1.5 CPU FP32（CLS+L2与官方query instruction）。Encbank为j=12、w=0、512-token chunks、residual-only。模型/BGE revision、adapter SHA、全部样本清单及完整数值见[summary.json](F:/Paper_Evolve/exp/encbank_bge_matched_20260918/summary.json)；原始记录见[quality/process_00/records.jsonl](F:/Paper_Evolve/exp/encbank_bge_matched_20260918/quality/process_00/records.jsonl)。

**核心结论：预定网格未达到±5%在线TTFT匹配，因此报告相邻工作点，不称严格等延迟。** 同BGE下，raw k6 / raw k8 / Encbank k12的测试TTFT为510.8 / 633.9 / 578.0ms，F1为4.55 / 5.17 / 11.81。这只支持本次Qasper、共享adapter协议下的结果，不替代原稿其他任务或完整冷E2E结论。

'''
tail='''
## D．独立适配的KV对照与未启动的条件项

已有CacheBlend-style独立LoRA/Encbank配对结果：LongEval 74.67/72.67、Qasper 4.25/11.37。4000 steps、训练tokens、rank/alpha、数据顺序等已核对；500题四臂共2000条的输入、选块顺序及预算对应。相同训练预算不等于FLOPs/GPU小时一致；不拿不同环境的质量运行耗时算速度比。核对记录见[KV对齐记录](F:/Paper_Evolve/exp/encbank_bge_matched_20260918/existing_analysis/kv_alignment.json)。

可选LoCoMo语义重评分尚未执行，本报告不填语义正确率或其区间。新的j训练扫描、生产并发、额外agent任务、lower-Write训练均未启动。原Qwen3.5/3.8的五benchmark量化及大LoRA评测属于已授权的独立在途工作，仍继续。

本地A三轮校准、正式质量生成及汇总均正常退出；600正式答案逐条decode、参考答案F1、排名/选块、完整分母检查通过，600预热不计样本。全部600正式答案自然达到128-token上限，空答0、错误0。配对区间按148个完整context簇重采样10000次，seed20260918。验证记录见[delivery_verification.json](F:/Paper_Evolve/exp/encbank_bge_matched_20260918/delivery_verification.json)。
'''
(root/'REVIEWER_RESULTS_zh.md').write_text(intro+a.replace('# 相同BGE的Qasper质量与在线TTFT：邻近工作点比较','## A．相同BGE的Qasper质量与在线TTFT：邻近工作点比较',1)+'\n'+b.replace('# 已有复用与更新日志重算','## B/C．已有复用与更新日志重算',1).replace('## 逐文档摊销','### 逐文档摊销',1).replace('## 更新维护（分项核算）','### 更新维护（分项核算）',1)+tail,encoding='utf-8')
print(root/'REVIEWER_RESULTS_zh.md')
