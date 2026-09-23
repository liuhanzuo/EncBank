"""Read-only error characterization after all planned benchmark cells finish."""
import collections,json,re
from pathlib import Path
from transformers import AutoTokenizer
R=Path(__file__).resolve().parent
rows=json.loads((R/'scored_items.json').read_text())
tok=AutoTokenizer.from_pretrained('/srv/encbank/comem_sparse_slurm_20260912/models/Qwen3-8B',local_files_only=True)
inputs={}
for cell in {x['cell'] for x in rows}:inputs.update({x['id']:x for x in json.loads((R/'inputs'/(cell+'.json')).read_text())})
summary={};examples={}
for task in ['single','multikey','vt']:
    subset=[x for x in rows if x['cell'].startswith(task)];counts=collections.Counter();examples[task]=[]
    for x in subset:
        counts['items']+=1
        for arm,stop in x['stops'].items():counts[arm+'_length_cap']+=int(stop=='length')
        if x['scores']['comem']!=1 or x['scores']['kd_selected']==1:continue
        counts['comem_full_correct_student_not_full']+=1
        case=inputs[x['id']];packed=tok.decode([t for chunk in case['selected_chunks'] for t in chunk])
        counts['all_references_present_in_retrieved_text']+=int(all(r in packed for r in x['references']))
        firstline=x['predictions']['kd_selected'].splitlines()[0]
        if task in ['single','multikey']:
            numbers=re.findall(r'\b\d{6,8}\b',firstline)
            if numbers:
                counts['first_line_has_number']+=1
                counts['first_number_is_wrong']+=int(numbers[0] not in x['references'])
                counts['wrong_first_number_is_in_retrieved_text']+=int(numbers[0] not in x['references'] and numbers[0] in packed)
        else:
            query=tok.decode(case['query_token_ids']);numbers=collections.Counter(re.findall(r'\b\d+\b',firstline))
            counts['first_line_repeats_query_number_at_least_3_times']+=int(any(n>=3 and v in query for v,n in numbers.items()))
        if len(examples[task])<3:
            examples[task].append(dict(cell=x['cell'],id=x['id'],references=x['references'],
                comem=x['predictions']['comem'],kd640=x['predictions']['kd_selected'],kd2048=x['predictions']['kd2048']))
    summary[task]=dict(counts)
(R/'failure_analysis.json').write_text(json.dumps(dict(summary=summary,examples=examples,
    caveat='Descriptive patterns, not a causal mechanism test. All models used the same fixed generation caps. Late answers beyond these caps were not evaluated.'),indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2))
