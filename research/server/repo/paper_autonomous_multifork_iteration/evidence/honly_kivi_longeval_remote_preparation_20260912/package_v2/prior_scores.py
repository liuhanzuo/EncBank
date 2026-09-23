"""Join already completed original scores; no tokenizer, model, or rescoring."""
from protocol import CONFIG, local, read, sha

def load_prior_scores(plan, ids):
    sources=plan['prior_four_arm_evidence']
    for spec in sources.values():assert sha(local(spec['path']))==spec['sha256']
    prior_plan=read(local(sources['plan']['path']))
    report=read(local(sources['independent_report']['path']))
    old=read(local(sources['original_analysis']['path']))
    assert report['status']=='PASS_independent_LongEval8K100_four_FP16_completion'
    assert prior_plan['configuration']==CONFIG==plan['configuration']
    assert prior_plan['fixture']==plan['fixture'] and prior_plan['labels']==plan['labels'] and prior_plan['scorer']==plan['scorer']
    assert prior_plan['backbone_dtype']==plan['backbone_dtype']=='float16'
    assert prior_plan['attention_backend']==plan['attention_backend']=='sdpa'
    assert prior_plan['activation']==plan['activation'] and prior_plan['model_root']==plan['model_root']
    assert prior_plan['natural_qa_adapter']==plan['natural_qa_adapter']
    assert prior_plan['arm_order']==['dense','h16','h8','h4']
    assert old['plan_sha256']==sources['plan']['sha256']
    assert old['status']=='complete_LongEval400_answers_400_Writes_2404_phases_pending_parent_shell_Slurm_and_independent_verification'
    assert set(old['arms'])==set(prior_plan['arm_order'])
    scores={}
    for arm in prior_plan['arm_order']:
        row=old['arms'][arm];values=row['per_item_numeric_exact']
        assert row['n']==row['documents']==row['Writes']==100 and row['phase_count']==601
        assert list(values)==ids and all(type(v) is int and v in (0,1) for v in values.values())
        scores[arm]=[values[x] for x in ids]
    return {'scores':scores,'original_contrasts':old['paired_contrasts'],
            'summary':{'source_bindings':sources,'scores_reused_without_rescoring':True,
                       'previous_original_analyzer_or_models_rerun':False,
                       'separate_acquisition_blocks':True,
                       'arms':{a:{k:old['arms'][a][k] for k in ('n','documents','numeric_exact_accuracy_percent','correct','zero_scores','raw_empty_predictions','stripped_empty_predictions','capped','EOS_stops')} for a in prior_plan['arm_order']}}}
