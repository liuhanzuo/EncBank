"""Summarize completed jobs and deliver only code, manifests and result summaries."""
import hashlib,json,statistics,subprocess,sys,tarfile
from pathlib import Path
R=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
assert Path('/srv/encbank') in R.parents
jobs=json.loads((R/'submissions.json').read_text());brief=[];accounting=[]
for job in jobs:
    out=R/job['arm']/'results';receipt=json.loads((out/'parent_exit.json').read_text())
    assert receipt['actual_wait'] and receipt['returncode']==0,(job,receipt)
    s=json.loads((out/'summary.json').read_text());assert s['complete']
    row=subprocess.check_output(['sacct','-X','-j',job['job'],'--format=JobID,State,ExitCode,Start,End,NodeList','--noheader','-P'],text=True)
    assert 'COMPLETED|0:0' in row,row
    accounting.append(row)
    b=dict(arm=job['arm'],job=job['job'],depths=s.get('depths'),test_mean=s.get('test_mean'),head_MiB=s.get('head_parameter_bytes',0)/2**20)
    if 'runtime' in s:
        b['runtime12']=[dict(mode=r['mode'],ms=r['median_ms_per_token'],history_MiB=r['history_persistent_bytes']/2**20,
            projection_ms=statistics.median(p['wall_ms'] for p in r['projection']) if r['projection'] else None) for r in s['runtime'] if r['chunks']==12]
    if 'validation' in s:b['selected_validation_kl']=min(r['validation_kl'] for r in s['validation'])
    if 'distillation' in s:b['distillation']=s['distillation']
    if (out/'replay.json').exists():
        b['replay']=[{k:v for k,v in r.items() if k not in ['observations','extra_anchor_prepare','checks']} for r in json.loads((out/'replay.json').read_text())]
    if job['arm']=='hot8':b['hot_buffer']=s
    brief.append(b)
(R/'slurm_accounting.txt').write_text(''.join(accounting))
(R/'compact_results.json').write_text(json.dumps(brief,indent=2,ensure_ascii=False)+'\n')
files=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json','.txt'] and p.name not in ['dataset.json','delivery_manifest.json']]
for job in jobs:files.extend((R/job['arm']/'results').glob('*.json'))
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
(R/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(R/'results_for_local.tar.gz','w:gz') as tar:
    for name in list(manifest)+['delivery_manifest.json']:tar.add(R/name,arcname=name)
print(json.dumps(brief,indent=2,ensure_ascii=False))
