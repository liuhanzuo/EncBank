"""CPU-only official 2WikiMQA scoring, reached only after four real worker exits."""
from pathlib import Path
import argparse, ast, collections, datetime, random, re, statistics, string
from protocol import ARMS, local, read, save, sha, document_groups, token_sha

def metric_functions(plan):
    scope = {'re': re, 'string': string, 'Counter': collections.Counter}
    specs = [(plan['scorer'], {'normalize_answer','f1_score','qa_f1_score'}),
             (plan['official_eval'], {'scorer'})]
    for binding, names in specs:
        path = local(binding['path'])
        assert sha(path) == binding['sha256']
        nodes = [n for n in ast.parse(path.read_text(encoding='utf-8')).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == names
        if 'scorer' in names: scope['dataset2metric'] = {'2wikimqa': scope['qa_f1_score']}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope['qa_f1_score'], scope['scorer']

def main():
    p=argparse.ArgumentParser()
    for key in ('plan','expected-plan-sha256','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();assert sha(a.plan)==a.expected_plan_sha256 and not Path(a.output).exists()
    plan=read(a.plan)
    for name,digest in plan['source_sha256'].items():assert sha(local(name))==digest,name
    for key in ('fixture','labels','scorer','official_eval'):
        assert sha(local(plan[key]['path']))==plan[key]['sha256'],key
    fixture=read(local(plan['fixture']['path']));groups=document_groups(fixture)
    labels=read(local(plan['labels']['path']));assert labels['fixture_sha256']==plan['fixture']['sha256']
    label_map={x['id']:x for x in labels['items']}
    ids=[x['id'] for x in fixture['items']];assert len(ids)==200 and set(ids)==set(label_map)
    qa_f1,official_scorer=metric_functions(plan)
    reports={};scores={}
    for arm in ARMS:
        out=local(plan['outputs'][arm]);execution=read(out/'execution.json');worker=read(out/'worker.json');result=read(out/'result.json')
        assert execution['worker_exit_code']==0 and execution['actual_parent_wait'] and execution['process_exit_observed'] and execution['source_stable']
        assert execution['status']=='completed_pending_independent_analysis'
        assert worker['status']=='completed' and worker['scientific_result_sha256']==sha(out/'result.json')
        assert result['outer_closed'] and all(x['completed'] for x in result['phases'])
        phases=collections.Counter(x['name'] for x in result['phases'])
        assert phases['document']==200 and phases['outer_model_load_quality_and_release']==1
        rows=result['rows'];docs=result['documents'];assert [r['id'] for r in rows]==ids and len(docs)==200
        assert sorted(r['execution_index'] for r in rows)==list(range(200))
        for i,(row,item) in enumerate(zip(rows,fixture['items'])):
            assert row['source_index']==i and row['document_id']==item['document_id']
            assert row['document_token_sha256']==token_sha(item['document_token_ids']) and row['query_token_sha256']==token_sha(item['query_token_ids'])
            assert row['full_prompt_with_BOS_token_sha256']==item['full_prompt_with_BOS_token_sha256']
            tokens=row['generated_token_ids'];assert 1<=len(tokens)<=32 and 151645 not in tokens
            assert row['hit_generation_cap']==(len(tokens)==32) and row['stopped_on_eos']==(len(tokens)<32)
        assert [d['document_id'] for d in docs]==[g[0] for g in groups]
        for doc,(_,indices) in zip(docs,groups):
            assert doc['source_indices']==indices and doc['item_ids']==[ids[i] for i in indices]
            assert doc['completed_queries']==len(indices)
        if arm!='dense':
            assert len(result['phases'])==1201 and phases['Write']==200 and phases['entry_release']==200
            assert phases['Store_verification_before']==200 and phases['Store_verification_after']==200 and phases['Read_selection_dequant_query_and_decode']==200
            assert all(d['H_entry_writes']==1 and d['entry_unchanged_all_queries'] and d['entry_release']['all_tracked_tensor_objects_released'] for d in docs)
            assert all(d['store']['document_lower_KV_bytes']==0 for d in docs)
            assert all(r['entry_unchanged'] and r['selected_read_released'] and r['dequantized_chunk_indices']==r['selected_chunk_indices'] and len(r['selected_chunk_indices'])<=12 for r in rows)
        else:
            assert len(result['phases'])==401 and phases['Dense_full_context_prefill_and_decode']==200 and phases['Write']==0
        predictions=[r['prediction'] for r in rows];refs=[label_map[x]['references'] for x in ids]
        scores[arm]=[max(qa_f1(pred,ref,all_classes=label_map[item_id]['all_classes']) for ref in references)
                     for item_id,pred,references in zip(ids,predictions,refs)]
        official=official_scorer('2wikimqa',predictions,refs,[])
        assert official==round(100*statistics.fmean(scores[arm]),2)
        reports[arm]={'n':200,'documents':200,'official_QA_F1_percent':official,
            'unrounded_mean_F1_percent':100*statistics.fmean(scores[arm]),
            'per_item_F1':dict(zip(ids,scores[arm])),'zero_scores':sum(s==0 for s in scores[arm]),
            'raw_empty_predictions':sum(not s for s in predictions),'capped':sum(r['hit_generation_cap'] for r in rows),
            'H_Writes':sum(d['H_entry_writes'] for d in docs),'phase_count':len(result['phases']),
            'result_sha256':sha(out/'result.json'),'worker_sha256':sha(out/'worker.json'),'execution_sha256':sha(out/'execution.json')}
    rng=random.Random(20260911);draws=[[rng.randrange(200) for _ in range(200)] for _ in range(10000)]
    contrasts={}
    for arm,ref in [('h8','h16'),('h4','h16'),('h16','dense'),('h8','dense'),('h4','dense')]:
        delta=[100*(x-y) for x,y in zip(scores[arm],scores[ref])]
        sums=[sum(delta[i] for i in indices) for _,indices in groups];sizes=[len(indices) for _,indices in groups]
        values=sorted(sum(sums[g] for g in draw)/sum(sizes[g] for g in draw) for draw in draws)
        contrasts[arm+'_minus_'+ref]={'mean_points':statistics.fmean(delta),'descriptive_95_percentile_interval':[values[249],values[9749]],'quantization_isolation':ref=='h16'}
    record={'status':'complete_full_2WikiMQA800_answers_600_H_Writes_plus_Dense200prefills_pending_parent_and_independent_verification',
        'plan_sha256':a.expected_plan_sha256,'finished_at':datetime.datetime.now().astimezone().isoformat(),
        'arms':reports,'paired_contrasts':contrasts,
        'resampling':{'RNG':'Python Random MT19937','seed':20260911,'draws':10000,'unit':'200 exact-document clusters sampled with replacement, pooled item-weighted mean; one question per document','zero_based_endpoint_indices':[249,9749]},
        'scope':'Complete 2WikiMQA200 only, no multi-query reuse coverage, not six-task LongBench macro, no timing or capacity ranking; scalarEOS151645 and first-step EOS suppression inherited from the qualified CoMem decoder.',
        'limitations':['Each exact document has one question; no multi-query reuse coverage; Dense unchanged full prompt per question','H three arms share custom LoRA, Dense LoRA off','No truncation or gold-selected retrieval','Intervals including0 do not establish equivalence or noninferiority']}
    save(a.output,record);print({k:v['official_QA_F1_percent'] for k,v in reports.items()})

if __name__=='__main__':main()
