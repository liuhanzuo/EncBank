"""BABILong_qa1_1k_100 grouped entries and variable-length QA; imports no model at module load."""
import gc, importlib.util, sys, time
from pathlib import Path
from protocol import ROOT, local, save, token_sha, document_groups, sha

def fingerprint(entry):
    import hashlib, torch
    h = hashlib.sha256()
    for name, tensor in entry.tensor_items():
        h.update(name.encode()); h.update(str((tuple(tensor.shape),str(tensor.dtype),str(tensor.device))).encode())
        h.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()

def load_method(plan):
    path = local(plan['tokenwise_runtime_root'])/'tokenwise_runtime.py'
    assert sha(path) == plan['source_sha256'][path.relative_to(ROOT).as_posix()]
    spec = importlib.util.spec_from_file_location('_longeval_fp16_frozen_tokenwise', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path
    return module.Method

def run(model, tokenizer, fixture, arm, *, plan, output, provenance, profiler, natural_read, kernels=None):
    from phase_profile import weak_tensor_refs, release_report
    import torch
    Method = load_method(plan)
    method = Method(model, tokenizer, "h16", resume_j=12, reader_binding=plan['reader_binding'],
                    bos_token_id=151643, eos_token_id=151645, kernels=kernels)
    assert arm=='comem_frozen_j12' and method.memory.engine.resume_j==12 and method.memory.bits==16
    assert not any('lora_' in n for n,_ in model.named_parameters())
    result = {'schema':'BABILong_qa1_1k_100_FP16_frozen_CoMem_j12_remote_v1','status':'running','arm':arm,
        'configuration':plan['configuration'],'provenance':provenance,'rows':[],'documents':[],
        'phases':profiler.records,'complete_quality_evidence':False,
        'scope':'New independent frozen CoMem j12 LoRA OFF qa1_1k100; exact original inputs and cap20; native FP16 SDPA natural quality diagnostics; not a published checkpoint or fixed-work infra claim.'}
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
                            'source_id':item['source_id'],'source_ordinal':item['source_ordinal'],
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
                        row['retrieval_configuration'] = ({'selector':'iter_bm25','topk':12,'iter_hop_topk':4,
                            'iter_rounds':0,'chunk_size':512} if arm=='comem_frozen_j12' else None)
                        group_rows.append(row); completed[index]=row; execution_index+=1
                        result['rows']=[completed[i] for i in sorted(completed)]; save(output,result)
                        print(f'{arm}: {execution_index}/100 id={item["id"]} tokens={row["generation_length"]}',flush=True)
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
        assert len(result['rows'])==100 and len(result['documents'])==plan['documents_per_arm']
        assert sum(d['entry_writes'] for d in result['documents'])==plan['documents_per_arm']
        result['status']='answers_complete_outer_and_process_exit_pending';return result
    except BaseException as error:
        result.update(status='failed_partial_not_complete_quality',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        method.close(); del method
        result['phases']=profiler.records;save(output,result)
