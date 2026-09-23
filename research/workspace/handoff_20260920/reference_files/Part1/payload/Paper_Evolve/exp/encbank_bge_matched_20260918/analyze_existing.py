"""Recompute requested document-level cost statistics from original completed logs."""
import collections,json,statistics
import config

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]

def main():
    out=config.ROOT/'existing_analysis';out.mkdir(exist_ok=True)
    source=config.OLD/'reuse_update_attempt2'
    receipts=load(config.OLD/'queue_attempt2/receipts.json')
    receipt=next(r for r in receipts if r['cell']=='reuse_update')
    assert receipt['actual_parent_wait'] and receipt['actual_exit_code']==0
    complete=load(source/'complete.json');assert complete['requests']==2400 and complete['updates']==40
    rows=lines(source/'requests.jsonl');assert len(rows)==2400 and all(r['status']=='ok' for r in rows)
    docs=sorted({r['document_id'] for r in rows});assert len(docs)==10
    reports=[]
    for doc in docs:
        arm={a:sorted([r for r in rows if r['document_id']==doc and r['arm']==a],key=lambda r:r['stream_index']) for a in ['raw','encbank']}
        assert all(len(v)==80 and [r['stream_index'] for r in v]==list(range(80)) for v in arm.values())
        assert [r['id'] for r in arm['raw']]==[r['id'] for r in arm['encbank']]
        store=load(source/(doc+'_store.json'));write=store['write_seconds']
        assert all(r['cold_write_seconds']==write for r in arm['encbank'])
        cumulative={}
        for a,rs in arm.items():
            total=write if a=='encbank' else 0.;vals=[]
            for r in rs:total+=r['online_seconds'];vals.append(total)
            cumulative[a]=vals
        gains=[cumulative['encbank'][i]<cumulative['raw'][i] for i in range(80)]
        crossover=next((i+1 for i in range(80) if all(gains[i:])),None)
        reports.append(dict(document_id=doc,initial_write_s=write,raw_Q10_s=cumulative['raw'][9],
            encbank_Q10_s=cumulative['encbank'][9],raw_Q80_s=cumulative['raw'][79],encbank_Q80_s=cumulative['encbank'][79],
            gain_Q10=gains[9],gain_Q80=gains[79],sustained_crossover_within_80=crossover,
            output_tokens={a:statistics.mean(r['generated_tokens'] for r in rs) for a,rs in arm.items()},cumulative_s=cumulative))
    crosses=[r['sustained_crossover_within_80'] for r in reports if r['sustained_crossover_within_80'] is not None]
    summary={k:statistics.mean(r[k] for r in reports) for k in ['initial_write_s','raw_Q10_s','encbank_Q10_s','raw_Q80_s','encbank_Q80_s']}
    summary.update(gain_Q10_documents=sum(r['gain_Q10'] for r in reports),gain_Q80_documents=sum(r['gain_Q80'] for r in reports),
        crossover_documents=len(crosses),crossover_median=statistics.median(crosses) if crosses else None,
        crossover_range=[min(crosses),max(crosses)] if crosses else None,
        output_tokens={a:statistics.mean(r['generated_tokens'] for r in rows if r['arm']==a) for a in ['raw','encbank']})
    updates=lines(source/'updates.jsonl');assert len(updates)==40 and all(r['full_state_equal'] for r in updates)
    maintenance=[]
    for change in ['append','replace','insert','delete']:
        rs=[r for r in updates if r['change']==change];assert len(rs)==10
        inc=statistics.mean(r['incremental_write_seconds'] for r in rs)
        full=statistics.mean(r['full_rebuild_seconds'] for r in rs)
        index=statistics.mean(r['full_BM25_rebuild_search_seconds'] for r in rs)
        maintenance.append(dict(change=change,n=10,incremental_write_s=inc,index_rebuild_plus_one_search_s=index,
            incremental_accounted_s=inc+index,full_write_s=full,full_accounted_s=full+index,
            ratio_incremental_over_full=(inc+index)/(full+index),speedup=(full+index)/(inc+index),H_equal=10))
    payload=dict(complete=True,new_gpu=False,source=str(source),receipt=receipt,reuse=summary,documents=reports,maintenance=maintenance,
        boundaries=dict(reuse='Pretokenized existing documents; both arms online include same observed BM25 rebuild/search. Encbank adds once-only measured H Write. No separately persisted BM25 index; common input tokenization/tensor setup excluded symmetrically.',
            maintenance='Component-accounted sum, NOT continuous wall-clock E2E. Both paths use the SAME observed full BM25 rebuild+one search component; no incremental BM25 index implementation. Includes chunk tensor preparation in that CPU region. Excludes shared input tokenization, durable storage IO, metadata/version transaction, model load. No updated-fact QA claim.'))
    (out/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    md=['# 已有复用与更新日志重算','',f"来源：{source}；2400请求、40更新及真实父进程wait0已核对。无新LLM运行。",'',
        '## 逐文档摊销','', '| 文档 | 初始H Write秒 | Raw Q10秒 | Encbank Q10秒 | Raw Q80秒 | Encbank Q80秒 | 持续crossover | Q80获益 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in reports:md.append('| '+r['document_id']+' | '+' | '.join(f"{r[k]:.3f}" for k in ['initial_write_s','raw_Q10_s','encbank_Q10_s','raw_Q80_s','encbank_Q80_s'])+f" | {r['sustained_crossover_within_80'] or '80问内未出现'} | {'是' if r['gain_Q80'] else '否'} |")
    md+=['',f"平均初始Write {summary['initial_write_s']:.3f}秒；Q10 raw/Encbank {summary['raw_Q10_s']:.3f}/{summary['encbank_Q10_s']:.3f}秒，{summary['gain_Q10_documents']}/10文档获益。Q80为{summary['raw_Q80_s']:.3f}/{summary['encbank_Q80_s']:.3f}秒，{summary['gain_Q80_documents']}/10获益。",
        f"80问内持续crossover出现于{len(crosses)}/10文档；中位位置{summary['crossover_median']}，范围{summary['crossover_range']}。只对已观察80问成立。",
        f"平均输出tokens raw/Encbank {summary['output_tokens']['raw']:.2f}/{summary['output_tokens']['encbank']:.2f}；自然生成工作量不同，不能称固定输出加速。",
        '', '两侧均计入原请求的BM25全量重建+search；没有另一个未计入的持久检索索引。Encbank只额外加一次实际测得的H Write。共同token化、初始CPU chunk构造均在边界外。', '',
        '## 更新维护（分项核算）','', '| 更新 | 增量Write秒 | 全量索引重建+一次查询秒 | 增量合计秒 | 全重建合计秒 | 增量/全量 | H一致 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    labels=dict(append='追加1024 tokens',replace='替换1 token',insert='插入2 tokens',delete='删除3 tokens')
    for r in maintenance:md.append('| '+labels[r['change']]+' | '+' | '.join(f"{r[k]:.4f}" for k in ['incremental_write_s','index_rebuild_plus_one_search_s','incremental_accounted_s','full_accounted_s','ratio_incremental_over_full'])+' | 10/10 |')
    md+=['','这是独立阶段之和，不是连续wall-clock E2E。现有40项均保存BM25重建与一次search的合并计时，因此不需要再跑CPU索引计时；该合并值共同加到两个路径，不能称“增量BM25维护”。原40/40 H一致性保持。',
        '未计入输入token化、持久化磁盘IO、版本/元数据事务、模型加载；不主张改事实后的QA已验证。']
    (out/'RESULTS_zh.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(dict(reuse=summary,maintenance=maintenance),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
