"""Paired benchmark scoring helpers; inference never imports this module."""
import hashlib,json
from pathlib import Path
import numpy as np
from official_scoring import string_match_all
R=Path(__file__).resolve().parent
def load_phase(phase):
    configs=json.loads((R/'configs.json').read_text());data={};outputs={}
    for run,c in configs.items():
        if c['phase']!=phase:continue
        out=R/run/'results';receipt=json.loads((out/'parent_exit.json').read_text())
        assert receipt['actual_wait'] and receipt['returncode']==0,(run,receipt)
        summary=json.loads((out/'summary.json').read_text())
        p=out/'predictions.jsonl';assert hashlib.sha256(p.read_bytes()).hexdigest()==summary['predictions_sha256']
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        expected=json.loads(Path(c['input']).read_text())[c['start']:c['end']]
        assert [x['id'] for x in rows]==[x['id'] for x in expected]
        assert len(rows)==c['end']-c['start'] and len({x['id'] for x in rows})==len(rows)
        labels=json.loads((R/'scoring_only'/(c['cell']+'.json')).read_text())
        arms=summary['arms'];matrix=[]
        for row in rows:
            refs=labels[row['id']];assert refs
            matrix.append([sum(float(ref.lower() in row['arms'][arm]['prediction'].lower()) for ref in refs)/len(refs) for arm in arms])
        matrix=np.array(matrix)
        for j,arm in enumerate(arms):
            official=string_match_all([row['arms'][arm]['prediction'] for row in rows],[labels[row['id']] for row in rows])
            # Match the official Python summation and rounding order at x.xx5 ties.
            assert official==round(sum(matrix[:,j].tolist())/len(matrix)*100,2)
            assert abs(official-float(matrix[:,j].mean()*100))<=.00500001
        data[c['cell']]=dict(arms=arms,matrix=matrix)
        outputs[c['cell']]=[dict(id=row['id'],index=row['index'],references=labels[row['id']],
            scores={arm:float(matrix[i,j]) for j,arm in enumerate(arms)},
            predictions={arm:row['arms'][arm]['prediction'] for arm in arms}) for i,row in enumerate(rows)]
    return data,outputs
def summarize(arrays,arms):
    a=np.concatenate(arrays,axis=0);mean=a.mean(0)*100;base=arms.index('native')
    rng=np.random.default_rng(20260921)
    boot=np.mean([x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1) for x in arrays],axis=0)*100
    return dict(items=len(a),score={arm:round(sum(a[:,j].tolist())/len(a)*100,2) for j,arm in enumerate(arms)},
        score_unrounded={arm:float(mean[j]) for j,arm in enumerate(arms)},
        vs_native={arm:dict(delta_pp=float(mean[j]-mean[base]),
            paired95ci_pp=np.quantile(boot[:,j]-boot[:,base],[.025,.975]).tolist(),
            lower_score_items=int((a[:,j]<a[:,base]).sum()),higher_score_items=int((a[:,j]>a[:,base]).sum()),
            native_full_correct_to_not_full=int(((a[:,base]==1)&(a[:,j]<1)).sum())) for j,arm in enumerate(arms)})
def groups(data):
    arms=next(iter(data.values()))['arms'];assert all(x['arms']==arms for x in data.values())
    result={cell:summarize([x['matrix']],arms) for cell,x in data.items()}
    tasks={task:summarize([x['matrix'] for cell,x in data.items() if cell.startswith(task)],arms) for task in ['single','multikey','vt']}
    return dict(cells=result,tasks=tasks,all=summarize([x['matrix'] for x in data.values()],arms))
