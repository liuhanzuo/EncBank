import json
from pathlib import Path
for pid in [4058393,4058511,4059594,4135114,4135169,4135390]:
    d = Path('/proc') / str(pid)
    row = dict(pid=pid)
    for name in ['cmdline','stat','environ']:
        try:
            text = (d / name).read_bytes().decode(errors='replace')
            if name == 'environ':
                env = dict(e.split('=',1) for e in text.split('\0') if '=' in e)
                text = {k:v for k,v in env.items() if k in ['SLURM_JOB_ID','SLURM_JOBID','APPTAINER_CACHEDIR','TMPDIR','APPTAINER_CONTAINER']}
            row[name] = text
        except OSError as error:
            row[name] = repr(error)
    row['owner'] = d.stat().st_uid if d.exists() else None
    print(json.dumps(row))
