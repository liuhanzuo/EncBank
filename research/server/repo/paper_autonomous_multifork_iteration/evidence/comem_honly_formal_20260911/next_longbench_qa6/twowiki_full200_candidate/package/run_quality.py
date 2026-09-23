"""Natural greedy quality only. No labels, no training and no runtime offload."""
import gc, hashlib, sys, time
from pathlib import Path
from protocol import save, ROOT

sys.path.insert(0, str(ROOT / 'paper_autonomous_multifork_iteration/evidence/comem_honly_formal_20260911/runtime'))
from honly import HOnlyMemory

def entry_fingerprint(entry):
    # Short-lived diagnostic byte copies are destroyed here, never inference state.
    h = hashlib.sha256()
    for name, tensor in entry.tensor_items():
        h.update(name.encode())
        h.update(str((tuple(tensor.shape), str(tensor.dtype), str(tensor.device))).encode())
        h.update(tensor.detach().contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes())
    return h.hexdigest()

def dense_generate(model, ids, *, eos_token_id, max_new_tokens):
    import torch
    device = next(model.parameters()).device
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    output = model(input_ids=tokens, use_cache=True, logits_to_keep=1, return_dict=True)
    cache = output.past_key_values
    logits = output.logits[0, -1].float()
    logits[eos_token_id] = float('-inf')
    generated = [int(logits.argmax().item())]
    del output, logits, tokens
    stopped = False
    for _ in range(max_new_tokens - 1):
        step = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        output = model(input_ids=step, past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
        cache = output.past_key_values
        next_id = int(output.logits[0, -1].float().argmax().item())
        del output, step
        if next_id == eos_token_id:
            stopped = True
            break
        generated.append(next_id)
    del cache
    return {'generated_token_ids': generated, 'stopped_on_eos': stopped,
            'hit_generation_cap': not stopped, 'read_len': len(ids),
            'selected_chunk_indices': None, 'selected_read_released': True}

def run(model, tokenizer, fixture, arm, *, configuration, output, provenance, profiler):
    import torch
    from protocol import ROOT, document_groups, token_sha
    sys.path.insert(0, str(ROOT / 'gpu/kvquant_quality_balanced'))
    from phase_profile import weak_tensor_refs, release_report
    bits = None if arm == 'dense' else int(arm[1:])
    memory = None if bits is None else HOnlyMemory(model, tokenizer,
        resume_j=12, bits=bits, group_size=64, reader_binding=provenance['reader_binding'],
        bos_token_id=151643, eos_token_id=151645)
    cap = configuration['max_new_tokens']
    assert cap == 32
    result = {'schema': 'honly_full_2wikimqa200_quality_v1', 'status': 'running', 'arm': arm,
        'configuration': configuration, 'provenance': provenance, 'rows': [], 'documents': [],
        'phases': profiler.records, 'complete_quality_evidence': False,
        'scope': 'Complete 2WikiMQA200/200 exact-document groups, cap32; no multi-query reuse coverage, six-task macro or infra.'}
    save(output, result)
    completed_rows = {}
    execution_index = 0
    for group_index, (document_id, indices) in enumerate(document_groups(fixture)):
        first = fixture['items'][indices[0]]
        entry = None
        doc = {'document_id': document_id, 'item_ids': [fixture['items'][i]['id'] for i in indices],
               'source_indices': indices, 'document_token_sha256': token_sha(first['document_token_ids']),
               'document_tokens': len(first['document_token_ids']), 'H_entry_writes': int(memory is not None)}
        group_rows = []
        with profiler.phase('document', document_id=document_id):
            if memory is not None:
                with profiler.phase('Write', document_id=document_id):
                    entry = memory.encode_ids(first['document_token_ids'], document_id=document_id)
                with profiler.phase('Store_verification_before', document_id=document_id):
                    doc['store'] = entry.inventory()
                    before = entry_fingerprint(entry)
                    references = weak_tensor_refs(entry.tensor_items())
                    doc['entry_hash_before'] = before
            else:
                doc['store'] = None
            for index in indices:
                item = fixture['items'][index]
                started = time.monotonic()
                row = {'id': item['id'], 'document_id': document_id, 'source_index': index,
                    'source_id': item['source_id'], 'source_ordinal': item['source_ordinal'],
                    'execution_index': execution_index, 'dataset': item['dataset'],
                    'document_tokens': len(item['document_token_ids']), 'query_tokens': len(item['query_token_ids']),
                    'document_token_sha256': doc['document_token_sha256'],
                    'query_token_sha256': token_sha(item['query_token_ids']),
                    'full_prompt_with_BOS_token_sha256': item['full_prompt_with_BOS_token_sha256']}
                if memory is not None:
                    with profiler.phase('Read_selection_dequant_query_and_decode', item_id=item['id'], document_id=document_id):
                        generated = memory.generate_ids(entry, item['query_token_ids'],
                            bm25_query_ids=item['bare_question_token_ids'], max_new_tokens=cap)
                    with profiler.phase('Store_verification_after', item_id=item['id'], document_id=document_id):
                        after = entry_fingerprint(entry)
                        row.update(entry_hash_before=before, entry_hash_after=after, entry_unchanged=before == after)
                        assert before == after and generated['selected_read_released']
                else:
                    with profiler.phase('Dense_full_context_prefill_and_decode', item_id=item['id'], document_id=document_id):
                        generated = dense_generate(model, [151643] + item['document_token_ids'] + item['query_token_ids'],
                            eos_token_id=151645, max_new_tokens=cap)
                row.update(generated)
                row['raw_generated_text'] = tokenizer.decode(generated['generated_token_ids'], skip_special_tokens=False)
                row['prediction'] = tokenizer.decode(generated['generated_token_ids'], skip_special_tokens=True)
                row['elapsed_seconds_descriptive_not_fixed_work_timing'] = time.monotonic() - started
                group_rows.append(row)
                completed_rows[index] = row
                execution_index += 1
                result['rows'] = [completed_rows[i] for i in sorted(completed_rows)]
                save(output, result)
                print(f'{arm}: {execution_index}/200 id={item["id"]} tokens={len(row["generated_token_ids"])}', flush=True)
            if memory is not None:
                with profiler.phase('entry_release', document_id=document_id):
                    entry.release()
                    del entry
                    gc.collect()
                    doc['entry_release'] = release_report(references)
                    assert doc['entry_release']['all_tracked_tensor_objects_released']
                doc['entry_unchanged_all_queries'] = all(x['entry_unchanged'] for x in group_rows)
            else:
                doc['Dense_full_context_prefills'] = len(indices)
            doc['completed_queries'] = len(group_rows)
        result['documents'].append(doc)
        save(output, result)
    assert len(result['rows']) == 200 and len(result['documents']) == 200
    assert sum(x['H_entry_writes'] for x in result['documents']) == (0 if arm == 'dense' else 200)
    result['status'] = 'answers_complete_outer_and_process_exit_pending'
    save(output, result)
    del memory
    return result
