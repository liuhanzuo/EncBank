"""Saved-output official LongBench QA F1; require all six successful native exits first."""
from pathlib import Path
import argparse, ast, collections, datetime, random, re, statistics, string
from protocol import ARMS, CONFIG, HERE, local, read, save, sha, document_groups, token_sha

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
        if 'scorer' in names: scope['dataset2metric'] = {'musique': scope['qa_f1_score']}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope['qa_f1_score'], scope['scorer']

def held(record, field='actual_exit_code'):
    assert record[field] == 0 and record['actual_parent_wait'] and record['process_exit_observed']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','expected-plan-sha256','output'):p.add_argument('--'+key,required=True)
    args=p.parse_args();assert sha(args.plan)==args.expected_plan_sha256 and not Path(args.output).exists()
    plan=read(args.plan);assert plan['configuration']==CONFIG and plan['arm_order']==list(ARMS)
    for name,digest in plan['source_sha256'].items():assert sha(local(name))==digest,name
    for key in ('fixture','labels','scorer','official_eval','input_coverage'):
        spec=plan[key];assert sha(local(spec['path']))==spec['sha256'],key
    batch=read(local(plan['batch_output'])/'batch_status.json')
    assert batch['plan_sha256']==args.expected_plan_sha256 and batch['status']=='all_one_arm_exited_complete_pending_analysis'
    closed={}
    stages=batch['stages'];assert [s['arm'] for s in stages]==list(ARMS)
    for arm,stage in zip(ARMS,stages):
        out=local(plan['outputs'][arm]);execution=read(out/'execution.json');worker=read(out/'worker.json')
        held(execution,'worker_exit_code')
        assert stage['actual_exit_code']==0 and stage['execution_sha256']==sha(out/'execution.json')
        assert execution['status']=='completed_pending_independent_analysis' and execution['worker_pid']==worker['pid']
        assert worker['status']=='completed' and worker['scientific_result_sha256']==sha(out/'result.json')
        assert worker['plan_sha256']==execution['plan_sha256']==args.expected_plan_sha256
        assert execution['physical_gpu_identity']==batch['physical_gpu_identity']==worker['physical_gpu_identity']
        closed[arm]=(out,execution,worker)
    # References/scorer are first opened only after every worker/transport has closed successfully.
    fixture=read(local(plan['fixture']['path']));groups=document_groups(fixture);items=fixture['items'];ids=[x['id'] for x in items]
    labels=read(local(plan['labels']['path']));assert labels['fixture_sha256']==plan['fixture']['sha256']
    label_map={x['id']:x for x in labels['items']};assert set(label_map)==set(ids) and len(ids)==200 and len(groups)==200
    qa_f1,official_scorer=metric_functions(plan);reports={};scores={}
    for arm in ARMS:
        out,execution,worker=closed[arm];result=read(out/'result.json');prov=result['provenance']
        assert result['status']=='worker_complete_parent_exit_pending' and result['outer_closed'] and result['configuration']==CONFIG
        assert prov['plan_sha256']==args.expected_plan_sha256 and prov['source_sha256']==plan['source_sha256'] and prov['fixture']==plan['fixture']
        if arm.startswith('kivi'):assert prov['actual_kivi_backend']['binary']==plan['kivi_backend']['binary'] and prov['actual_kivi_backend']['source_sha256']==plan['kivi_backend']['source_sha256'] and prov['actual_kivi_backend']['performance_backend']
        assert prov['backend_qualification']==plan['backend_qualification']
        assert worker['qualified_compute_capability']==plan['backend_qualification']['compute_capability']
        assert prov['backbone_dtype']=='torch.float16' and prov['attention_backend']=='sdpa'
        assert prov['adapter_active'] is False and prov['LoRA_parameter_dtypes']==[] and prov['configuration']['resume_j']==12
        assert prov['actual_BOS_token_id']==151643 and prov['EOS_token_id']==151645 and prov['first_step_EOS_suppressed'] and not prov['model_and_tokenizer_configs_mutated']
        assert Path(prov['resolved_quality_driver_file']).name=='run_quality.py' and Path(prov['resolved_natural_qa_file']).name=='natural_qa.py'
        rows=result['rows'];docs=result['documents'];phases=result['phases'];names=collections.Counter(x['name'] for x in phases)
        assert rows and [x['id'] for x in rows]==ids and len(docs)==200 and len(phases)==1201
        assert phases==worker['phases'] and all(x['completed'] for x in phases)
        assert names=={'outer_model_load_native_quality_and_release':1,'document':200,'Write':200,'Store_verification_before':200,'Read_complete_query_and_natural_decode':200,'Store_verification_after':200,'entry_release':200}
        assert [d['document_id'] for d in docs]==[g[0] for g in groups]
        for doc,(_,indices) in zip(docs,groups):
            assert doc['source_indices']==indices and doc['item_ids']==[ids[i] for i in indices] and doc['completed_queries']==len(indices)
            assert doc['entry_writes']==1 and doc['entry_unchanged_all_queries'] and doc['entry_release']['all_tracked_tensor_objects_released']
            if arm=='comem_frozen_j12':assert doc['store']['document_lower_KV_bytes']==0 and doc['store']['bits']==16 and doc['store']['resume_j']==12 and not doc['store']['native_hidden_or_KV_host_offload']
            else:assert doc['store']['prefix_tokens']==doc['document_tokens']+1 and not doc['store']['native_hidden_or_KV_host_offload']
        for index,(row,item) in enumerate(zip(rows,items)):
            assert row['source_index']==index and row['document_id']==item['document_id'] and row['dataset']=='musique'
            assert row['document_token_sha256']==token_sha(item['document_token_ids']) and row['query_token_sha256']==token_sha(item['query_token_ids']) and row['bare_question_token_sha256']==token_sha(item['bare_question_token_ids'])
            assert row['full_prompt_with_BOS_token_sha256']==item['full_prompt_with_BOS_token_sha256']
            assert row['entry_unchanged'] and row['entry_hash_before']==row['entry_hash_after'] and row['selected_read_released'] and row['request_release']['all_tracked_tensor_objects_released']
            tokens=row['generated_token_ids'];assert 1<=len(tokens)<=32 and 151645 not in tokens
            assert row['generation_length']==len(tokens) and row['max_new_tokens']==32
            assert row['hit_generation_cap']==(len(tokens)==32) and row['stopped_on_eos']==(len(tokens)<32)
            assert row['actual_BOS_token_id']==151643 and row['EOS_token_id']==151645 and row['first_step_EOS_suppressed']
            assert row['query_tokens']==row['query_prefill_calls']==len(item['query_token_ids']) and row['query_tokens_per_call']==1
            assert row['decode_forward_count']==(len(tokens) if row['stopped_on_eos'] else len(tokens)-1) and row['head_calls']==row['decode_forward_count']+1
            assert row['pending_token_id_before_request_release']==(None if row['stopped_on_eos'] else tokens[-1])
            selected=row['selected_chunk_indices']
            if arm=='comem_frozen_j12':
                assert selected==sorted(set(selected)) and 0<=len(selected)<=12 and all(0<=i<(len(item['document_token_ids'])+511)//512 for i in selected)
            else:assert selected is None
        for row in rows:
            t=row['timing'];n=len(row['generated_token_ids'])
            assert t['pre_read_cuda_synchronized'] and 0<=t['ttft_seconds']<=t['final_decision_seconds']<=t['read_through_request_cleanup_seconds']<=t['full_Read_including_text_decode_seconds']
            assert t['decode_tokens_first_to_last']==n-1
            if n==1:assert t['decode_seconds_first_to_last_non_EOS'] is None and t['decode_tokens_per_second'] is None
            else:
                seconds=t['decode_seconds_first_to_last_non_EOS'];assert seconds>=0
                assert t['decode_tokens_per_second']==((n-1)/seconds if seconds>0 else None)
            assert (t['EOS_decision_end_seconds'] is not None)==row['stopped_on_eos']
        assert sorted(r['execution_index'] for r in rows)==list(range(200))
        assert all(x['allowed'] for x in execution['admissions']) and execution['stable_interval_seconds']>=45
        assert worker['cuda_policy']['allocator_cap_bytes']==200*2**30
        assert all(x['allowed'] and x['cuda_free_bytes']>=220*2**30 for x in worker['pre_migration_checks'])
        assert max(p['peak_reserved_bytes'] for p in phases)<=200*2**30
        predictions=[r['prediction'] for r in rows];refs=[label_map[x]['references'] for x in ids]
        scores[arm]=[max(qa_f1(pred,ref,all_classes=label_map[item_id]['all_classes']) for ref in references) for item_id,pred,references in zip(ids,predictions,refs)]
        official=official_scorer('musique',predictions,refs,[]);assert official==round(100*statistics.fmean(scores[arm]),2)
        reports[arm]={'n':200,'documents':200,'official_QA_F1_percent':official,'unrounded_mean_F1_percent':100*statistics.fmean(scores[arm]),
            'per_item_F1':dict(zip(ids,scores[arm])),'zero_scores':sum(s==0 for s in scores[arm]),'raw_empty_predictions':sum(not s for s in predictions),
            'stripped_empty_predictions':sum(not s.strip() for s in predictions),'capped':sum(r['hit_generation_cap'] for r in rows),'EOS_stops':sum(r['stopped_on_eos'] for r in rows),
            'generation_lengths':[len(r['generated_token_ids']) for r in rows],'per_item_request_timing':{r['id']:r['timing'] for r in rows},'per_item_position_usage':{r['id']:r['position_usage'] for r in rows},'per_item_unscaled_regime':{r['id']:r['unscaled_input_regime'] for r in rows},'Writes':200,'phase_count':1201,
            'result_sha256':sha(out/'result.json'),'worker_sha256':sha(out/'worker.json'),'execution_sha256':sha(out/'execution.json')}
    assert plan['analysis']['contrasts']==[] and plan['analysis']['bootstrap_executed'] is False
    contrasts={}
    save(args.output,{'status':'complete_frozen_CoMem_j12_musique_200_answers_200_Writes_1201_phases_pending_parent_shell_Slurm_and_independent_verification',
        'plan_sha256':args.expected_plan_sha256,'finished_at':datetime.datetime.now().astimezone().isoformat(),'arms':reports,'paired_contrasts':contrasts,
        'resampling':plan['analysis'],'scope':'New frozen CoMem j12 LoRA OFF musique: exact 200 questions/200 documents, cap32, original LongBench QA F1, FP16 native SDPA. Natural timing only; no published checkpoint or fixed-work infrastructure claim.',
        'limits':['Frozen CoMem j12 uses no LoRA; new algorithm measurement, not a published checkpoint claim','Same native SDPA primitive policy is not identical CUDA dispatch or attention graph','Intervals including0 do not establish equivalence/noninferiority']})
    print({a:r['official_QA_F1_percent'] for a,r in reports.items()})

if __name__=='__main__':main()
