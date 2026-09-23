"""Root-invoked native sbatch route: manifest, singleton, duplicate, and fresh four-GPU gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
HOME = Path('/srv/encbank')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def command(argv):
    child = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if child.returncode:
        raise RuntimeError({'argv': argv, 'returncode': child.returncode, 'stderr': child.stderr})
    return {'argv': argv, 'returncode': child.returncode, 'stdout': child.stdout, 'stderr': child.stderr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--expected-manifest-sha256', required=True)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    HERE.relative_to(HOME)
    manifest_path = HERE / 'package_manifest.json'
    assert sha(manifest_path) == args.expected_manifest_sha256
    manifest = read(manifest_path)
    for item in manifest['files']:
        path = (HERE / item['path']).resolve()
        path.relative_to(HERE)
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], str(path)
    plan = read(HERE / 'plan.json')
    assert not Path(plan['output']).exists(), 'Existing output; no duplicate'
    validation = read(HERE / 'staged_validation.json')
    assert validation['status'] == 'PASS' and validation['plan_sha256'] == sha(HERE / 'plan.json')
    assert validation['package_manifest_sha256'] == args.expected_manifest_sha256
    if not args.submit:
        print(json.dumps({'status': 'FILES_AND_STAGED_VALIDATION_MATCH_NO_SUBMISSION'}))
        return
    # Retained lock: after any uncertain response the operator reads the receipt, never repeats sbatch.
    (HERE / 'submission.lock').mkdir(exist_ok=False)
    receipt_path = HERE / 'submission_receipt.json'
    receipt = {'status': 'FRESH_VALIDATION', 'started_at_epoch': time.time(), 'submission_invoked': False,
               'plan_sha256': sha(HERE / 'plan.json')}
    def save():
        temp = receipt_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(receipt, indent=2) + '\n')
        temp.replace(receipt_path)
    save()
    try:
        queue = command(['squeue', '--me', '--array', '--noheader', '--format=%i|%T|%j'])
        receipt['queue_before'] = queue
        owned = 0
        details = []
        for line in queue['stdout'].splitlines():
            jid, state, name = [x.strip() for x in line.split('|')]
            assert re.fullmatch(r'[0-9_]+', jid)
            detail = command(['scontrol', '--oneliner', 'show', 'job', jid])
            details.append(detail)
            text = detail['stdout']
            assert 'Command=' + str(HERE / 'batch.sbatch') not in text
            if 'qencbank' in name.lower() or '/qencbank_align_codex_20260911/' in text:
                assert name != plan['job_name'], 'Independent cell already active/pending'
                match = re.search(r'(?:^| )ReqTRES=([^ ]+)', text)
                assert match
                resources = dict(x.split('=', 1) for x in match.group(1).split(','))
                requested = int(resources.get('gres/gpu', '0'))
                assert requested == 1, 'Unexpected owned request; inspect instead of guessing'
                owned += requested
        assert owned + 1 <= 4, 'Four requested GPU limit includes running and pending'
        receipt.update(existing_job_details=details, owned_gpu_requests_before=owned,
                       owned_gpu_requests_after=owned + 1)
        assert not Path(plan['output']).exists()
        shell = (HERE / 'batch.sbatch').read_text()
        for required in ['#SBATCH --gres=gpu:1', '#SBATCH --dependency=singleton',
                         '#SBATCH --no-requeue', '#SBATCH --constraint=l20d',
                         '#SBATCH --exclude=gpu-node1,gpu-node2,gpu-node8']:
            assert required in shell
        receipt.update(status='SUBMITTING', submission_invoked=True)
        save()
        result = command(['sbatch', '--parsable', str(HERE / 'batch.sbatch')])
        receipt['sbatch'] = result
        job = result['stdout'].strip().split(';')[0]
        assert job.isdigit(), 'Ambiguous sbatch response; never resubmit'
        receipt.update(status='SLURM_SUBMITTED', job_id=job, submitted_at_epoch=time.time())
        save()
        print(json.dumps({'status': receipt['status'], 'job_id': job, 'receipt_sha256': sha(receipt_path)}))
    except BaseException as exc:
        receipt.update(status='FAILED_OR_UNCERTAIN_NO_RETRY' if receipt['submission_invoked'] else 'FAILED_BEFORE_SUBMISSION',
                       error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    main()
