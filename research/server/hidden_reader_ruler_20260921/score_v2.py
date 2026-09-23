"""Score all paired predictions only after generation, retaining item-level evidence."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from official_scoring import string_match_all
R=Path(__file__).resolve().parent
ARMS=['comem','comem_split','kd256','kd_selected','kd2048']
def save(name,o):(R/name).write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
configs=json.loads((R/'configs.json').read_text());inventory={x['cell']:x for x in json.loads((R/'input_inventory.json').read_text())}
rows=[];scores={};checks=[]
control_configs=json.loads((R/'numerical_control/configs.json').read_text())
for cell,c in configs.items():
    control_out=R/'numerical_control'/cell/'results'
    assert json.loads((control_out/'parent_exit.json').read_text())['returncode']==0
    assert json.loads((control_out/'status.json').read_text())['phase']=='COMPLETE'
    controls=[json.loads(line) for line in (control_out/'predictions.jsonl').read_text().splitlines()]
    assert len(controls)==100 and len({x['id'] for x in controls})==100
    assert sha(control_out/'predictions.jsonl')==json.loads((control_out/'summary.json').read_text())['predictions_sha256']
    by_id={x['id']:x for x in controls}
    out=R/cell/'results' if control_configs[cell]['control_only'] else control_out
    assert json.loads((out/'parent_exit.json').read_text())['returncode']==0
    assert json.loads((out/'status.json').read_text())['phase']=='COMPLETE'
    summary=json.loads((out/'summary.json').read_text());assert sha(out/'predictions.jsonl')==summary['predictions_sha256']
    assert sha(R/'scoring_only'/(cell+'.json'))==inventory[cell]['labels_sha256']
    labels=json.loads((R/'scoring_only'/(cell+'.json')).read_text())
    predictions=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
    if control_configs[cell]['control_only']:
        for x in predictions:
            control=by_id[x['id']]
            for key in ['source_document_sha256','selected_indices','memory_tokens','query_tokens']:assert x[key]==control[key],(cell,x['id'],key)
            x['arms']['comem_split']=control['arms']['comem_split']
    assert len(predictions)==len({x['id'] for x in predictions})==100
    assert set(labels)=={x['id'] for x in predictions}
    cell_scores=[]
    for x in predictions:
        ref=labels[x['id']];assert ref
        values={arm:sum(float(r.lower() in x['arms'][arm]['prediction'].lower()) for r in ref)/len(ref) for arm in ARMS}
        cell_scores.append([values[a] for a in ARMS])
        rows.append(dict(cell=cell,id=x['id'],references=ref,scores=values,predictions={a:x['arms'][a]['prediction'] for a in ARMS},
            stops={a:x['arms'][a]['stop_reason'] for a in ARMS},selected_indices=x['selected_indices'],memory_tokens=x['memory_tokens']))
    arr=np.array(cell_scores);scores[cell]=arr
    for j,arm in enumerate(ARMS):
        official=string_match_all([x['arms'][arm]['prediction'] for x in predictions],[labels[x['id']] for x in predictions])
        assert abs(official-round(float(arr[:,j].mean()*100),2))<1e-6,(cell,arm,official)
    checks.append(dict(cell=cell,items=100,official_score_verified=True,exact_cache_control=json.loads((control_out/'exact_cache_control.json').read_text())))
def describe(arrays):
    joined=np.concatenate(arrays,0);rng=np.random.default_rng(20260921)
    # Stratified paired item bootstrap, preserving each task/length cell's weight.
    boots=np.mean([x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1) for x in arrays],axis=0)*100
    means=joined.mean(0)*100
    result=dict(items=len(joined),score={a:round(float(means[j]),4) for j,a in enumerate(ARMS)},comparison={})
    for j,arm in enumerate(ARMS[1:],1):
        d=boots[:,j]-boots[:,0]
        result['comparison'][arm]=dict(delta_vs_comem_pp=round(float(means[j]-means[0]),4),
            paired_95ci_pp=np.quantile(d,[.025,.975]).tolist(),
            lower_score_items=int((joined[:,j]<joined[:,0]).sum()),higher_score_items=int((joined[:,j]>joined[:,0]).sum()),
            comem_full_correct_to_student_not_full=int(((joined[:,0]==1)&(joined[:,j]<1)).sum()),
            student_full_correct_from_comem_not_full=int(((joined[:,0]<1)&(joined[:,j]==1)).sum()),
            delta_vs_kd256_pp=round(float(means[j]-means[2]),4),
            delta_vs_comem_split_pp=round(float(means[j]-means[1]),4),
            paired_95ci_vs_split_pp=np.quantile(boots[:,j]-boots[:,1],[.025,.975]).tolist(),
            paired_95ci_vs_kd256_pp=np.quantile(boots[:,j]-boots[:,2],[.025,.975]).tolist())
    return result
result=dict(checkpoints=json.loads((R/'checkpoints.json').read_text()),cells={c:describe([x]) for c,x in scores.items()},
    groups={g:describe([x for c,x in scores.items() if predicate(c)]) for g,predicate in [
        ('official_niah_only',lambda c:not c.startswith('vt')),('official_niah_single',lambda c:c.startswith('single')),
        ('official_niah_multikey',lambda c:c.startswith('multikey')),('comem_ruler_style_vt',lambda c:c.startswith('vt')),('all_three_tasks_exploratory',lambda c:True)]},
    bootstrap='10,000 paired item bootstrap draws stratified by task/length; no multiplicity adjustment; describes this frozen sample, not all RULER',
    source_verification=[])
manifest=json.loads((R/'source_manifest.json').read_text())
for rel,expected in manifest.items():
    actual=sha(R/rel);assert actual==expected,(rel,expected,actual)
result['source_verification']=dict(checked_files=len(manifest),all_unchanged=True)
control_manifest=json.loads((R/'numerical_control/source_manifest.json').read_text())
for rel,expected in control_manifest.items():assert sha(R/'numerical_control'/rel)==expected,rel
result['source_verification']['control_files']=len(control_manifest)
save('scored_items.json',rows);save('score_checks.json',checks);save('scores.json',result)
jobs=json.loads((R/'submissions.json').read_text())+json.loads((R/'numerical_control/submissions.json').read_text());ids=','.join(x['job'] for x in jobs)
accounting=subprocess.check_output(['sacct','-j',ids,'--format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS,AllocTRES%60','--parsable2'],text=True)
(R/'accounting.txt').write_text(accounting)
print(json.dumps(result['groups'],indent=2))
