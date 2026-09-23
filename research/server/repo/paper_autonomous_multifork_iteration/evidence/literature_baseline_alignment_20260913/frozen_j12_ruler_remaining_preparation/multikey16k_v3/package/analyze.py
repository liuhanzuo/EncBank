"""Saved-output official NVIDIA RULER multikey recall; require one successful native exits first."""
from pathlib import Path
import argparse, ast, collections, datetime, random, re, statistics, string
from protocol import ARMS, CONFIG, HERE, local, read, save, sha, document_groups, token_sha

def metric_functions(plan):
    import importlib.util
    path=local(plan['scorer']['path']);assert sha(path)==plan['scorer']['sha256']
    spec=importlib.util.spec_from_file_location('_ruler_official_saved_scorer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    post=local(plan['postprocess']['path']);assert sha(post)==plan['postprocess']['sha256']
    node=next(n for n in ast.parse(post.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='postprocess_pred')
    scope={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),str(post),'exec'),scope)
    return lambda prediction,references:module.string_match_all([scope['postprocess_pred'](prediction,{})],[references])/100

def held(record, field='actual_exit_code'):
    assert record[field] == 0 and record['actual_parent_wait'] and record['process_exit_observed']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','expected-plan-sha256','output'):p.add_argument('--'+key,required=True)
    args=p.parse_args();assert sha(args.plan)==args.expected_plan_sha256 and not Path(args.output).exists()
    plan=read(args.plan);assert plan['configuration']==CONFIG and plan['arm_order']==list(ARMS)
    for name,digest in plan['source_sha256'].items():assert sha(local(name))==digest,name
    for key in ('fixture','labels','scorer'):
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
    label_map={x['id']:x for x in labels['items']};assert set(label_map)==set(ids) and len(ids)==100 and len(groups)==100
    string_match=metric_functions(plan);reports={};scores={}
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
        assert rows and [x['id'] for x in rows]==ids and len(docs)==100 and len(phases)==601
        assert phases==worker['phases'] and all(x['completed'] for x in phases)
        assert names=={'outer_model_load_native_quality_and_release':1,'document':100,'Write':100,'Store_verification_before':100,'Read_complete_query_and_natural_decode':100,'Store_verification_after':100,'entry_release':100}
        assert [d['document_id'] for d in docs]==[g[0] for g in groups]
        for doc,(_,indices) in zip(docs,groups):
            assert doc['source_indices']==indices and doc['item_ids']==[ids[i] for i in indices] and doc['completed_queries']==len(indices)
            assert doc['entry_writes']==1 and doc['entry_unchanged_all_queries'] and doc['entry_release']['all_tracked_tensor_objects_released']
            if arm=='comem_frozen_j12':assert doc['store']['document_lower_KV_bytes']==0 and doc['store']['bits']==16 and doc['store']['resume_j']==12 and not doc['store']['native_hidden_or_KV_host_offload']
            else:assert doc['store']['prefix_tokens']==doc['document_tokens']+1 and not doc['store']['native_hidden_or_KV_host_offload']
        for index,(row,item) in enumerate(zip(rows,items)):
            assert row['source_index']==index and row['document_id']==item['document_id'] and row['dataset']=='ruler_niah_multikey_1_16k'
            assert row['document_token_sha256']==token_sha(item['document_token_ids']) and row['query_token_sha256']==token_sha(item['query_token_ids']) and row['bare_question_token_sha256']==token_sha(item['bare_question_token_ids'])
            assert row['full_prompt_with_BOS_token_sha256']==item['full_prompt_with_BOS_token_sha256']
            assert row['entry_unchanged'] and row['entry_hash_before']==row['entry_hash_after'] and row['selected_read_released'] and row['request_release']['all_tracked_tensor_objects_released']
            tokens=row['generated_token_ids'];assert 1<=len(tokens)<=48 and 151645 not in tokens
            assert row['generation_length']==len(tokens) and row['max_new_tokens']==48
            assert row['hit_generation_cap']==(len(tokens)==48) and row['stopped_on_eos']==(len(tokens)<48)
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
        assert sorted(r['execution_index'] for r in rows)==list(range(100))
        assert all(x['allowed'] for x in execution['admissions']) and execution['stable_interval_seconds']>=45
        assert worker['cuda_policy']['allocator_cap_bytes']==200*2**30
        assert all(x['allowed'] and x['cuda_free_bytes']>=220*2**30 for x in worker['pre_migration_checks'])
        assert max(p['peak_reserved_bytes'] for p in phases)<=200*2**30
        predictions=[r['prediction'] for r in rows];refs=[label_map[x]['references'] for x in ids]
        assert all(isinstance(x,list) and len(x)==1 and all(isinstance(v,str) for v in x) for x in refs)
        scores[arm]=[100*string_match(pred,ref) for pred,ref in zip(predictions,refs)]
        official=statistics.fmean(scores[arm])
        reports[arm]={'n':100,'documents':100,'official_string_match_all_percent':official,
            'per_item_scores':dict(zip(ids,scores[arm])),'zero_scores':sum(s==0 for s in scores[arm]),
            'raw_empty_predictions':sum(not s for s in predictions),'stripped_empty_predictions':sum(not s.strip() for s in predictions),
            'capped':sum(r['hit_generation_cap'] for r in rows),'EOS_stops':sum(r['stopped_on_eos'] for r in rows),
            'generation_lengths':[len(r['generated_token_ids']) for r in rows],
            'per_item_request_timing':{r['id']:r['timing'] for r in rows},'Writes':100,'phase_count':601,
            'result_sha256':sha(out/'result.json'),'worker_sha256':sha(out/'worker.json'),'execution_sha256':sha(out/'execution.json')}
    assert plan['analysis']['contrasts']==[] and plan['analysis']['bootstrap_executed'] is False
    contrasts={}
    save(args.output,{'status':'complete_frozen_CoMem_j12_RULER_multikey16k_100_answers_100_Writes_601_phases_pending_parent_shell_Slurm_and_independent_verification',
        'plan_sha256':args.expected_plan_sha256,'finished_at':datetime.datetime.now().astimezone().isoformat(),'arms':reports,'paired_contrasts':contrasts,
        'resampling':plan['analysis'],'scope':'New independent frozen CoMem j12 LoRA OFF multikey16k100; exact original inputs and cap48; native FP16 SDPA natural quality diagnostics; not a published checkpoint or fixed-work infra claim.',
        'limits':['Frozen CoMem j12 uses no LoRA; this is a new adapter-free algorithm comparison, not a published checkpoint claim','Same native SDPA primitive policy is not identical CUDA dispatch or attention graph','Intervals including0 do not establish equivalence/noninferiority']})
    print({a:r['official_string_match_all_percent'] for a,r in reports.items()})

if __name__=='__main__':main()
