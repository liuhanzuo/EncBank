"""Opaque official final-state scoring, after all actor tasks end; aggregate only."""
import argparse,contextlib,io,json,os,shutil,zipfile
from pathlib import Path
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();R=Path(a.run).resolve();R.relative_to(H.resolve())
source=H/'appworld_public';dest=R/'official_scoring';dest.mkdir(exist_ok=False)
os.environ.update(APPWORLD_ROOT=str(dest),APPWORLD_CACHE=str(dest/'cache'),CUDA_VISIBLE_DEVICES='-1')
manifest=json.loads((H/'public_manifest.json').read_text())
for f in manifest['files']:
    rel=Path(f['path']);assert 'ground_truth' not in rel.parts
    target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/rel,target)
rows=json.loads((R/'actor_results.json').read_text())['rows'];result=[]
for row in rows:
    experiment=row['experiment_name']
    shutil.copytree(source/'experiments/outputs'/experiment,dest/'experiments/outputs'/experiment)
with open(os.devnull,'w',encoding='utf-8') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
    from appworld.common.constants import PASSWORD,SALT
    from appworld.common.crypto import decrypt_bytes
    from appworld.evaluator import evaluate_task
    bundle=H/'data-0.2.0.bundle'
    with zipfile.ZipFile(io.BytesIO(decrypt_bytes(bundle.read_bytes(),PASSWORD,SALT))) as z:
        for row in rows:
            task=row['task_id']
            for leaf in ('metadata.json','public_data.json','private_data.json','answer.json','evaluation.py','test_data.json'):
                name=f'data/tasks/{task}/ground_truth/{leaf}'
                target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
            tracker=evaluate_task(task_id=task,experiment_name=row['experiment_name'],suppress_errors=True,save_report=False)
            result.append(dict(task_id=task,success=bool(tracker.success),pass_count=tracker.pass_count,fail_count=tracker.fail_count,num_tests=tracker.num_tests))
record=dict(status='complete',rows=result,successes=sum(x['success'] for x in result),tasks=len(result),reference_solutions_read=False,rich_reports_emitted=False)
(R/'official_scores.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
