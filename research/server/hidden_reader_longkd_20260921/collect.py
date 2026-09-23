"""Compact longer-KD findings; model/checkpoint files remain remote."""
import hashlib,json,statistics,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
prior=R.parent/'hidden_reader_four_distill_20260921/fixed_lr1e5/results/summary.json'
for arm in ['kd256','cosine3e6','cosine1e5']:
    p=prior if arm=='kd256' else R/arm/'results/summary.json'
    x=json.loads(p.read_text());t=x['distillation']
    if arm!='kd256':assert json.loads((p.parent/'parent_exit.json').read_text())['returncode']==0
    records.append(dict(arm=arm,training=t,checkpoint_sha256=x['checkpoint_sha256'],
        old_test={k:statistics.mean(z[k] for z in x['test']) for k in ['kl','top1_agreement','student_nll','teacher_nll']},
        fresh_test={k:statistics.mean(z['trained'][k] for z in x['fresh_test']) for k in ['kl','top1_agreement','student_nll','teacher_nll']}))
manifest=json.loads((R/'source_manifest.json').read_text())
for rel,expected in manifest.items():assert sha(R/rel)==expected,rel
obj=dict(records=records,source_verification=dict(checked_files=len(manifest),all_unchanged=True))
(R/'compact_results.json').write_text(json.dumps(obj,indent=2)+'\n')
jobs=json.loads((R/'submissions.json').read_text())
(R/'accounting.txt').write_text(subprocess.check_output(['sacct','-j',','.join(j['job'] for j in jobs),'--format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS,AllocTRES%60','--parsable2'],text=True))
for x in records:print(json.dumps(x))
