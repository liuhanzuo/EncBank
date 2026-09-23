"""Posthoc output-length diagnostics: never relabel prefixes as independently run cap32."""
def prefix_diagnostic(generated,decode_ids):
    ids=list(generated['generated_token_ids'][:32])
    return {'schema':'posthoc_first32_of_new_cap64_v1','generated_token_ids':ids,
            'raw_generated_text':decode_ids(ids,skip_special_tokens=False),
            'prediction':decode_ids(ids,skip_special_tokens=True),'length':len(ids),
            'tokens_after_prefix':len(generated['generated_token_ids'])-len(ids),
            'is_independent_cap32_run':False,'semantic_correctness_measured':False}

def changes(full,prefix):
    return {'token_sequence_changed':full['generated_token_ids']!=prefix['generated_token_ids'],
            'decoded_answer_changed':full['prediction']!=prefix['prediction'],
            'raw_text_changed':full['raw_generated_text']!=prefix['raw_generated_text'],
            'new_tokens_beyond_first32':len(full['generated_token_ids'])-len(prefix['generated_token_ids'])}
