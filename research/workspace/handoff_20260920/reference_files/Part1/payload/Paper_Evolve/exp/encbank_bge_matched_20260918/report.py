"""Quality and timing report; document-cluster bootstrap, no missing-to-zero conversion."""
import collections,json,statistics
import numpy as np
import config

def main():
    path=config.ROOT/'quality/process_00'
    receipt=json.loads((path/'parent_exit.json').read_text());assert receipt['actual_wait'] and receipt['returncode']==0
    complete=json.loads((path/'complete.json').read_text());assert complete['complete']
    rows=[json.loads(l) for l in (path/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==complete['records'];rows=[r for r in rows if not r['warmup']]
    assert len(rows)==complete['formal_records'] and all(r['status']=='ok' for r in rows)
    decision=json.loads((config.ROOT/'budget_decision.json').read_text());spec=json.loads((config.ROOT/'data/complete.json').read_text())
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['method'],r['k_max'])].append(r)
    summary={}
    for (method,k),rs in groups.items():
        assert len(rs)==200 and len({r['id'] for r in rs})==200
        summary[f'{method}_k{k}']=dict(method=method,k_max=k,n_questions=200,n_exact_context_clusters=len({r['context_sha256'] for r in rs}),
            actual_pack_tokens_mean=statistics.mean(r['pack_tokens'] for r in rs),actual_pack_tokens_median=statistics.median(r['pack_tokens'] for r in rs),
            TTFT_median_ms=statistics.median(r['ttft_ms'] for r in rs),F1=100*statistics.mean(r['score'] for r in rs),
            output_tokens_mean=statistics.mean(r['generated_tokens'] for r in rs),length_cap_rate=statistics.mean(r['length_cap_reached'] for r in rs),
            empty_answer_count=sum(not r['prediction'].strip() for r in rs),error_count=0)
    cm={r['id']:r for r in groups[('encbank',12)]};comparisons=[]
    for k in decision['quality_raw_ks']:
        raw={r['id']:r for r in groups[('raw',k)]};assert set(raw)==set(cm)
        clusters=collections.defaultdict(list)
        for ident in raw:
            assert raw[ident]['context_sha256']==cm[ident]['context_sha256']
            clusters[raw[ident]['context_sha256']].append(100*(cm[ident]['score']-raw[ident]['score']))
        keys=sorted(clusters);sums=np.array([sum(clusters[k]) for k in keys]);counts=np.array([len(clusters[k]) for k in keys])
        rng=np.random.default_rng(config.SEED);sample=rng.integers(0,len(keys),size=(10000,len(keys)))
        boot=sums[sample].sum(1)/counts[sample].sum(1)
        ratio=summary[f'raw_k{k}']['TTFT_median_ms']/summary['encbank_k12']['TTFT_median_ms']
        comparisons.append(dict(raw_k=k,encbank_k=12,latency_ratio=ratio,latency_percent_difference=100*(ratio-1),
            delta_F1=summary['encbank_k12']['F1']-summary[f'raw_k{k}']['F1'],paired_cluster_CI95=np.percentile(boot,[2.5,97.5]).tolist(),
            test_matches_within_5_percent=abs(ratio-1)<=.05,calibration_matches=decision['strict_match_on_calibration'],
            bootstrap_clusters=len(keys),bootstrap_seed=config.SEED,bootstrap_samples=10000))
    value=dict(complete=True,protocol=spec,decision=decision,arms=summary,comparisons=comparisons,
        calibration_ID_manifest=str(config.ROOT/'data/calibration_ID_manifest.json'),evaluation_ID_manifest=str(config.ROOT/'data/evaluation_ID_manifest.json'),
        scope='Shared BGE CPU-FP32; online TTFT excludes document H bank/index construction and decode. New query-independent offline boundary; old same-k scores are a distinct protocol. No budget retuning on evaluation data.')
    (config.ROOT/'summary.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    md=['# 相同BGE的Qasper质量与在线TTFT：邻近工作点比较','',f"20个额外Qasper训练文档用于校准，200题/{spec['evaluation_contexts']}个完整context簇用于质量；完整context哈希、规范化文本和开头64词检查隔离。",'',
        '| 方法 | 最大k | 实际pack tokens 均值/中位 | TTFT中位 ms | F1 | 输出tokens均值 | 封顶率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key in [f'raw_k{k}' for k in decision['quality_raw_ks']]+['encbank_k12']:
        x=summary[key]
        md.append(f"| {'Raw replay（共享adapter）' if x['method']=='raw' else 'Encbank（共享adapter）'} | {x['k_max']} | {x['actual_pack_tokens_mean']:.1f}/{x['actual_pack_tokens_median']:.1f} | {x['TTFT_median_ms']:.1f} | {x['F1']:.2f} | {x['output_tokens_mean']:.2f} | {x['length_cap_rate']:.1%} |")
    md+=['','三臂各200题，合计600条正式答案；各臂空答和错误均为0。pack对Encbank表示缓存H与在线query组成的序列位置数，并不表示重放原文token。所有答案均在自然EOS协议下达到128-token上限；未设置min_new_tokens或强制固定输出长度。']
    for x in comparisons:md+=['',f"raw k={x['raw_k']}：TTFT比值 {x['latency_ratio']:.4f}，差{x['latency_percent_difference']:+.2f}%；Encbank−raw F1 {x['delta_F1']:+.2f}，配对context簇95%区间 [{x['paired_cluster_CI95'][0]:+.2f}, {x['paired_cluster_CI95'][1]:+.2f}]。测试集满足±5%：{'是' if x['test_matches_within_5_percent'] else '否'}。"]
    md+=['','预定校准网格和最终测试均未达到±5%：以上是邻近工作点，不能称严格等延迟。预算只由留出校准TTFT决定，不按测试F1或测试延迟重调。',
        '在线TTFT包含真实BGE query embedding、检索、fetch、query Write、prefill至第一个token；文档bank和BGE索引提前准备、成本单列，不是冷E2E等延迟。',
        f"旧协议有{spec['affected_old_questions']}题将部分问题token放入离线块；新实验保持完整prompt/token IDs、权重和解码不变，把Question前完整块作为bank、其余全部在线。旧4.82/11.81分数保留为不同边界的历史同k实验，不能混用。",
        '原始数据与所需全部字段见summary.json；H Write逐输入成本见quality/process_00/write.jsonl，BGE索引构建/复用记录见data/rankings.json；复用已有索引没有本批新建计时，不计为零成本。既有复用和维护结果见existing_analysis/RESULTS_zh.md。']
    (config.ROOT/'RESULTS_zh.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
