"""Collect result text and status only; exclude model weights, stores and optimizer state."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import tarfile

ROOT = Path('/data/liuhanzuo/encbank_v2_20260908')


if __name__ == '__main__':
    scripts = ROOT / 'workspace/exp/encbank_v2_benchmarks_20260908'
    sys.path.insert(0, str(scripts))
    from summarize_runs import summarize, read_json
    cpu_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
               'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}
    os.environ.update(cpu_env)
    # Keep scoring and tokenizer-based evidence work away from local GPU timings.
    progress = ROOT / 'outputs/report_progress'
    progress.mkdir(parents=True, exist_ok=True)
    for name, plan_name, output in [
            ('main', 'full_plan.json', 'benchmarks_v2'),
            ('trained_pub', 'trained_pub_full_plan.json', 'trained_pub'),
            ('cacheblend16', 'cacheblend16_full_plan.json', 'cacheblend16'),
            ('ruler_matched16k', 'ruler_matched16k_full_plan.json', 'ruler_matched16k'),
            ('oracle_support', 'oracle_support_full_plan.json', 'oracle_support')]:
        report = summarize(read_json(scripts / plan_name), ROOT / 'outputs' / output / 'full',
                           ROOT / 'workspace' if name == 'main' else None)
        (progress / (name + '_progress.json')).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    diagnostic_errors, created, attempted, pending = [], 0, 0, 0
    for receipt in (ROOT / 'outputs').rglob('COMPLETED.json'):
        run = receipt.parent
        if 'full' not in run.parts or not (run / 'run_config.json').exists():
            continue
        config = read_json(run / 'run_config.json')
        if config.get('benchmark') == 'locomo':
            script, result_name = 'diagnose_locomo_evidence.py', 'evidence_diagnostic.json'
        elif config.get('benchmark') == 'longbench' and 'hotpotqa' in config.get('options', {}).get('tasks', []):
            script, result_name = 'diagnose_gold_support.py', 'gold_support_diagnostic.json'
        else:
            continue
        if (run / result_name).exists() or not (scripts / script).exists():
            continue
        if attempted >= 2:
            pending += 1
            continue
        attempted += 1
        try:
            result = subprocess.run([sys.executable, str(scripts / script), '--run', str(run),
                '--model', str(ROOT / 'models/Qwen3-8B'), '--out', str(run / result_name)],
                env=cpu_env, capture_output=True, text=True, timeout=180)
            if result.returncode:
                diagnostic_errors.append({'run': str(run), 'error': result.stderr[-1200:]})
            else:
                created += 1
        except (OSError, subprocess.SubprocessError) as exc:
            diagnostic_errors.append({'run': str(run), 'error': str(exc)})
    diagnostics = ROOT / 'outputs/diagnostics'
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / 'evidence_collection_status.json').write_text(json.dumps({
        'status': 'failed' if diagnostic_errors else ('pending' if pending else 'completed'),
        'updated_at': datetime.now(timezone.utc).isoformat(), 'execution': 'remote CPU only',
        'attempted_missing_diagnostics': attempted, 'max_diagnostics_per_collection': 2,
        'pending_missing_diagnostics': pending,
        'created_missing_diagnostics': created, 'errors': diagnostic_errors}, indent=2) + '\n', encoding='utf-8')
    pairing = scripts / 'summarize_oracle_support.py'
    if pairing.exists() and (scripts / 'oracle_support_full_plan.json').exists():
        diagnostics = ROOT / 'outputs/diagnostics'
        diagnostics.mkdir(parents=True, exist_ok=True)
        status = {'updated_at': datetime.now(timezone.utc).isoformat(), 'purpose': 'CPU accuracy pairing only'}
        try:
            result = subprocess.run([sys.executable, str(pairing),
                '--oracle-root', str(ROOT / 'outputs/oracle_support/full'),
                '--main-root', str(ROOT / 'outputs/benchmarks_v2/full'),
                '--out', str(diagnostics / 'oracle_support_paired.json')],
                env=cpu_env,
                capture_output=True, text=True, timeout=90)
            status.update(status='completed' if result.returncode == 0 else 'failed',
                exit_code=result.returncode, output=result.stdout[-2000:], error=result.stderr[-2000:])
        except (OSError, subprocess.SubprocessError) as exc:
            status.update(status='failed', error=str(exc))
        (diagnostics / 'oracle_pairing_status.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    target = ROOT / 'outputs/report_payload.tar.gz'
    temp = target.with_suffix('.tmp')
    with tarfile.open(temp, 'w:gz') as archive:
        for folder in (ROOT / 'outputs', ROOT / 'logs'):
            for path in folder.rglob('*'):
                if path.is_file() and path.suffix in {'.json', '.jsonl', '.csv', '.log', '.md'}:
                    archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    temp.replace(target)
    print(target)
