"""Natural greedy quality only. No labels, no training and no runtime offload."""
import gc, hashlib, sys, time
from pathlib import Path
from protocol import save

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from honly import HOnlyMemory
from quality_timing import QualityTiming

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
    timing = QualityTiming(torch, device)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    output = model(input_ids=tokens, use_cache=True, logits_to_keep=1, return_dict=True)
    cache = output.past_key_values
    logits = output.logits[0, -1].float()
    logits[eos_token_id] = float('-inf')
    generated = [int(logits.argmax().item())]
    timing.sampled()
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
        timing.sampled()
    del cache
    quality_timing = timing.finish(len(generated))
    return {'generated_token_ids': generated, 'stopped_on_eos': stopped,
            'hit_generation_cap': not stopped, 'read_len': len(ids),
            'selected_chunk_indices': None, 'selected_read_released': True, 'quality_timing': quality_timing}

def run(model, tokenizer, fixture, arm, *, configuration, output, provenance, profiler):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'gpu/kvquant_quality_balanced'))
    from phase_profile import weak_tensor_refs, release_report
    bits = None if arm == 'dense' else int(arm[1:])
    memory = None if bits is None else HOnlyMemory(model, tokenizer,
        resume_j=12, bits=bits, group_size=64, reader_binding=provenance['reader_binding'],
        bos_token_id=151643, eos_token_id=151645)
    result = {'schema': 'honly_ruler_natural_quality_v1', 'status': 'running', 'arm': arm,
        'configuration': configuration, 'provenance': provenance, 'rows': [],
        'phases': profiler.records, 'complete_quality_evidence': False,
        'scope': 'One official RULER niah_multikey_1 max8192 cell, 100 new fixed questions; not fifteen-cell RULER macro or timing evidence.'}
    save(output, result)
    for item in fixture['items']:
        started = time.monotonic()
        entry = None
        row = {'id': item['id'], 'document_id': item['document_id'],
               'document_tokens': len(item['document_token_ids']), 'query_tokens': len(item['query_token_ids'])}
        with profiler.phase('document', item_id=item['id']):
            if memory is not None:
                with profiler.phase('Write', item_id=item['id']):
                    entry = memory.encode_ids(item['document_token_ids'], document_id=item['document_id'])
                with profiler.phase('Store_verification_before', item_id=item['id']):
                    row['store'] = entry.inventory()
                    before = entry_fingerprint(entry)
                    references = weak_tensor_refs(entry.tensor_items())
                with profiler.phase('Read_selection_dequant_query_and_decode', item_id=item['id']):
                    generated = memory.generate_ids(entry, item['query_token_ids'],
                        bm25_query_ids=item['bare_question_token_ids'], max_new_tokens=48)
                with profiler.phase('Store_verification_after', item_id=item['id']):
                    after = entry_fingerprint(entry)
                    row.update(entry_hash_before=before, entry_hash_after=after, entry_unchanged=before == after)
                    assert before == after
                with profiler.phase('entry_release', item_id=item['id']):
                    entry.release()
                    del entry
                    gc.collect()
                    row['entry_release'] = release_report(references)
                    assert row['entry_release']['all_tracked_tensor_objects_released']
            else:
                with profiler.phase('Dense_full_context_prefill_and_decode', item_id=item['id']):
                    generated = dense_generate(model, [151643] + item['document_token_ids'] + item['query_token_ids'],
                        eos_token_id=151645, max_new_tokens=48)
                row['store'] = None  # This quality pass does not claim resident Dense Store/timing.
            row.update(generated)
            row['raw_generated_text'] = tokenizer.decode(generated['generated_token_ids'], skip_special_tokens=False)
            row['prediction'] = tokenizer.decode(generated['generated_token_ids'], skip_special_tokens=True)
            row['elapsed_seconds_descriptive_not_fixed_work_timing'] = time.monotonic() - started
        result['rows'].append(row)
        save(output, result)
        print(f'{arm}: {len(result["rows"])}/100 id={item["id"]} tokens={len(row["generated_token_ids"])}', flush=True)
    assert len(result['rows']) == 100
    result['status'] = 'answers_complete_outer_and_process_exit_pending'
    save(output, result)
    del memory
    return result
