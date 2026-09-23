import json
import os
from pathlib import Path
import sys
import time
here = Path(__file__).resolve().parent
plan = json.loads((here / 'plan.json').read_text(encoding='utf-8-sig'))
out = Path(plan['output']).resolve()
out.relative_to(Path('/srv/encbank'))
if out.is_dir():
    record = {'actual_python_worker_exit_code': int(sys.argv[1]), 'parent_wait_observed': True,
              'observed_at_epoch': time.time(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'scientific_success_requires_complete_execution_and_all1986_predictions': True}
    (out / 'shell_exit.json').write_text(json.dumps(record, indent=2) + '\n')
