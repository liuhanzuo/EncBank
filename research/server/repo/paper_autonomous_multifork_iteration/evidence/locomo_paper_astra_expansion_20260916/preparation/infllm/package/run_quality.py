"""RULER single2 16K100 grouped entries and variable-length QA; imports no model at module load."""
import gc, importlib.util, sys, time
from pathlib import Path
from protocol import ROOT, local, save, token_sha, document_groups, sha, N, D, PHASES

def fingerprint(entry):
    import hashlib, torch
    h = hashlib.sha256()
    for name, tensor in entry.tensor_items():
        h.update(name.encode()); h.update(str((tuple(tensor.shape),str(tensor.dtype),str(tensor.device))).encode())
        h.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()

def load_method(plan):
    key = 'streaming_adapter' if 'streaming_adapter' in plan else 'infllm_adapter'
    path = local(plan[key]['path']); assert sha(path) == plan[key]['sha256']
    spec = importlib.util.spec_from_file_location('_bound_native_baseline',path)
    module = importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.Method


def run(model, tokenizer, fixture, arm, *, plan, output, provenance, profiler, natural_read, kernels=None):
    from phase_profile import weak_tensor_refs, release_report
    import torch
    Method = load_method(plan)
    method = Method(model, tokenizer, upstream_root=local(plan['infllm_upstream_root']), upstream_lock=local(plan['infllm_upstream_lock']['path']), write_chunk_size=2048)
    assert method.rope_base==1000000
    assert not any('lora_' in n for n,_ in model.named_parameters())
    result = {'schema':plan['schema'],'status':'running','arm':arm,
        'configuration':plan['configuration'],'provenance':provenance,'rows':[],'documents':[],
        'phases':profiler.records,'complete_quality_evidence':False,
        'adapter_active':False,'resume_j':None,'scope':plan['preparation_scope']}
    completed = {}; execution_index = 0; save(output,result)
    try:
        for document_id, indices in document_groups(fixture):
            first = fixture['items'][indices[0]]; entry = None; refs = None; group_rows = []
            doc = {'document_id':document_id,'source_indices':indices,
                'item_ids':[fixture['items'][i]['id'] for i in indices],
                'document_token_sha256':token_sha(first['document_token_ids']),
                'document_tokens':len(first['document_token_ids']),'entry_writes':0}
            result['active_document'] = doc; save(output,result)
            with profiler.phase('document', document_id=document_id):
                try:
                    with profiler.phase('Write', document_id=document_id):
                        entry = method.write(first['document_token_ids'], document_id)
                    doc['entry_writes'] = 1
                    with profiler.phase('Store_verification_before', document_id=document_id):
                        doc['store'] = entry.inventory(); before = fingerprint(entry)
                        refs = weak_tensor_refs(entry.tensor_items()); doc['entry_hash_before'] = before
                    for index in indices:
                        item = fixture['items'][index]; started = time.monotonic()
                        row = {'id':item['id'],'dataset':item['dataset'],'document_id':document_id,
                            'source_index':index,'execution_index':execution_index,
                            'source_id':item.get('source_id',item['id']),'source_ordinal':item.get('source_ordinal',index),
                            'document_tokens':len(item['document_token_ids']),
                            'document_token_sha256':doc['document_token_sha256'],
                            'query_token_sha256':token_sha(item['query_token_ids']),
                            'bare_question_token_sha256':token_sha(item['bare_question_token_ids']),
                            'full_prompt_with_BOS_token_sha256':item['full_prompt_with_BOS_token_sha256'],
                            'entry_hash_before':before}
                        with profiler.phase('Read_complete_query_and_natural_decode',item_id=item['id'],document_id=document_id):
                            generated = natural_read(method,entry,item['query_token_ids'],item['bare_question_token_ids'],
                                max_new_tokens=item['max_new_tokens'],bos_token_id=151643,eos_token_id=151645,
                                suppress_first_eos=True,decode_ids=tokenizer.decode,synchronize=torch.cuda.synchronize)
                        with profiler.phase('Store_verification_after',item_id=item['id'],document_id=document_id):
                            after = fingerprint(entry)
                            assert after == before and generated['request_release']['all_tracked_tensor_objects_released']
                        row.update(generated,entry_hash_after=after,entry_unchanged=after==before,
                            elapsed_seconds_descriptive_not_fixed_work_timing=time.monotonic()-started)
                        assert row['selected_chunk_indices'] is None
                        row['retrieval_configuration']=plan['method_configuration']
                        assert row['packed_read_tokens']==1+len(item['document_token_ids'])+len(item['query_token_ids'])
                        group_rows.append(row); completed[index]=row; execution_index+=1
                        result['rows']=[completed[i] for i in sorted(completed)]; save(output,result)
                        print(f'{arm}: {execution_index}/{N} id={item["id"]} tokens={row["generation_length"]}',flush=True)
                finally:
                    if entry is not None:
                        with profiler.phase('entry_release',document_id=document_id):
                            entry.release(); del entry; gc.collect()
                            if refs is not None:
                                doc['entry_release']=release_report(refs)
                                assert doc['entry_release']['all_tracked_tensor_objects_released']
                    doc['completed_queries']=len(group_rows)
                    doc['entry_unchanged_all_queries']=all(r['entry_unchanged'] for r in group_rows)
            result['documents'].append(doc); result.pop('active_document',None);save(output,result)
        assert len(result['rows'])==N and len(result['documents'])==D
        assert sum(d['entry_writes'] for d in result['documents'])==D
        result['status']='answers_complete_outer_and_process_exit_pending';return result
    except BaseException as error:
        result.update(status='failed_partial_not_complete_quality',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        method.close(); del method
        result['phases']=profiler.records;save(output,result)
