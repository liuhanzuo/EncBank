"""Check collected macro arithmetic and export complete four-benchmark results."""
import collections,csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BENCHMARKS={'ruler':(1500,15),'longeval':(500,5),'longbench':(1150,6),'babilong':(2100,21)}
LABELS={'cache_lora':'Encbank','cache_without_lora':'Encbank (without LoRA)',
 'replay_base':'Selected-text replay','replay_shared_lora':'Replay (shared Encbank LoRA)',
 'kvdirect':'KV-Direct (full source)','streamingllm':'StreamingLLM-style reference',
 'hcache_style':'HCache-style reference'}

def main():
    summary=json.loads((ROOT/'summary.json').read_text())
    assert summary['generation_verified_complete'] and summary['all_metrics_complete']
    results=[];cells=[];total=0;deltas={}
    for model,info in summary['models'].items():
        values=collections.defaultdict(list);seen=set();supports=collections.Counter()
        for shard in range(4):
            folder=ROOT/'results'/model/f'shard{shard}'
            verified=json.loads((folder/'verified_summary.json').read_text())
            assert verified['verified'] and verified['oom_records']==0
            assert verified['scope_id']==summary['scope']['scope_id']
            for line in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines():
                r=json.loads(line);key=(r['id'],r['arm'])
                assert key not in seen;seen.add(key)
                assert r['status']=='ok' and isinstance(r['score'],(float,int)) and 0<=r['score']<=1
                values[(r['arm'],r['benchmark'],r['task'],r['length'])].append(r['score'])
                supports[(r['arm'],r['benchmark'])]+=1
        assert len(seen)==36750 and len({uid for uid,arm in seen})==5250
        assert seen=={(uid,arm) for uid in {uid for uid,arm in seen} for arm in LABELS}
        total+=len(seen)
        for arm,label in LABELS.items():
            row=dict(model=model,j=info['j'],method=label,arm=arm)
            for benchmark,(expected_n,expected_cells) in BENCHMARKS.items():
                scores=[]
                for (a,b,task,length),vs in values.items():
                    if a==arm and b==benchmark:
                        score=100*sum(vs)/len(vs);scores.append(score)
                        cells.append(dict(model=model,j=info['j'],method=label,arm=arm,benchmark=b,
                            task=task,length=length,samples=len(vs),score=score))
                assert supports[(arm,benchmark)]==expected_n and len(scores)==expected_cells
                macro=sum(scores)/len(scores)
                assert math.isclose(macro,info['table'][arm][benchmark]['score'],abs_tol=1e-9)
                row[benchmark]=macro
            results.append(row)
        deltas[model]={b:info['table']['cache_lora'][b]['score']-info['table']['cache_without_lora'][b]['score'] for b in BENCHMARKS}
    assert total==73500
    out=ROOT/'delivery';out.mkdir(exist_ok=True)
    for filename,rows in [('four_benchmark_results.csv',results),('four_benchmark_cells.csv',cells)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'four_benchmark_check.json').write_text(json.dumps(dict(complete=True,records=total,oom=0,
        check='Collected support and cell-macro arithmetic match independently verified remote summaries',
        locomo_included=False,adaptation_gain_pp=deltas),indent=2)+'\n')
    lines=['# Qwen3.5 / Qwen3.8 前四项完整评测结果','',
        '截至 2026-09-17 02:15（北京时间）：两模型各 5,250 个输入、七组对照，共 73,500 条预测全部完成、逐条解码复评分通过，0 OOM。本文档不包含尚未完成的 LoCoMo 语义评分。','',
        '各指标按 0–100 报告。RULER 为 15 个任务/长度单元宏平均；LongEval 为五个长度宏平均；LongBench 为六个数据集 token F1 宏平均；BABILong 为 21 个任务/长度单元宏平均。','']
    for model,info in summary['models'].items():
        lines += [f'## {model}，j={info["j"]}','',
            '| 方法 | RULER | LongEval | LongBench F1 | BABILong |',
            '|---|---:|---:|---:|---:|']
        for row in results:
            if row['model']==model:lines.append('| '+row['method']+' | '+' | '.join(f'{row[b]:.2f}' for b in BENCHMARKS)+' |')
        lines += ['', 'Encbank 相比 without LoRA 的增益（依次对应表中四项，百分点）：'+
            '、'.join(f'{deltas[model][b]:+.2f}' for b in BENCHMARKS)+'。','']
    lines += ['## 如何解释','',
        'LoRA 在两个模型的四项评测上均改善 Encbank 的分数，支持适配有助于恢复缓存接口质量。它没有在全部任务上超过其他对照：例如 9B 的四项均低于全源 KV-Direct；27B 的 BABILong 高于全源 KV-Direct，而其他三项仍低于它。不能据此宣称跨任务一致的质量优势，也不能用本次质量实验推导速度优势。','',
        '本次为单一固定训练种子的结果，未做多种子显著性判断。depth 来自先前 exploratory screen，一些自然 QA 样本与 screen 重叠；不声称独立 held-out 最优深度。','',
        '两个模型均独立训练 4,000 步、16,384,000 token；采用各自原生 chat 模板并关闭 thinking。所有方法共享样本、输出预算和评分口径；检索路径使用 512-token chunk、iterative BM25 top-12。LongEval 所有方法统一最多生成 48 token。这是新模型评测组，与原 Qwen3-8B 的 plain-text/部分不同输出长度设置分开报告。','',
        'Selected-text replay 重放与 Encbank 相同的检索文本；shared LoRA 是将 Encbank adapter 用于 replay，不是另训 replay adapter。KV-Direct 读取完整 source。StreamingLLM-style 是 sink/recent 窗口重算参考；HCache-style 是不检索、不适配的独立 chunk hidden-state 参考，都不宣称复现原系统的全部实现。InfLLM/MemoryLLM 在这两种 backbone 上没有本次可用的兼容实现，因此不编造结果。','',
        '共享 Slurm 作业的耗时不作为 infra 测速；本批仅完成质量评测。当前论文尚未写入这些新分数。','',
        '## 文件','',
        '- `four_benchmark_results.csv`：两模型全部七组、四项宏平均。',
        '- `four_benchmark_cells.csv`：每个任务/长度单元的完整结果与样本数。',
        '- `four_verified_8shards.tar.gz`：原始 73,500 条预测、各分片协议、正确性检查、验证汇总及总汇总。',
        '- `four_benchmark_check.json`：本地完整支持集及宏平均算术核对。','',
        'LoCoMo 答案仍在生成。此前正式 Astra Judge 请求被服务返回 HTTP 403，保持未评分；不以 lexical F1 代替语义 Judge 分数。']
    (out/'FOUR_BENCHMARK_RESULTS_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(records=total,methods=len(results),cells=len(cells),adaptation_gain_pp=deltas)))

if __name__=='__main__':main()
