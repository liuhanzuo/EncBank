"""Static release checks only: no imports of model runtimes and no job submissions."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def main():
    inventory = json.loads((ROOT/'docs/source_inventory.json').read_text(encoding='utf-8'))
    errors = []
    checked = set()
    for row in inventory['files']:
        rel = row['published_path']
        path = (ROOT/rel).resolve()
        if not path.is_relative_to(ROOT):
            errors.append('Inventory path escapes repository: '+rel)
            continue
        if not path.is_file():
            errors.append('Missing source: '+rel)
        elif row.get('published_sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=row['published_sha256']:
            errors.append('Published source hash mismatch: '+rel)
        checked.add(rel)
    python_count = 0
    for name in ['encbank','train','eval','bench','runtime','tools']:
        if not (ROOT/name).is_dir():
            errors.append('Missing source directory: '+name)
        for path in (ROOT/name).rglob('*.py'):
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
                python_count += 1
            except (SyntaxError,UnicodeError) as exc:
                errors.append(f'{path.relative_to(ROOT)}: {exc}')
    forbidden={'.pt','.pth','.ckpt','.safetensors','.bin','.gguf','.onnx','.sif','.img','.zip','.gz'}
    credential=re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})\b')
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.parts or '__pycache__' in path.parts:
            continue
        if path.suffix in forbidden:
            errors.append('Forbidden artifact: '+str(path.relative_to(ROOT)))
        data=path.read_bytes()
        if b'\0' in data:
            errors.append('Unexpected binary: '+str(path.relative_to(ROOT)))
            continue
        if credential.search(data.decode('utf-8-sig')):
            errors.append('Credential pattern: '+str(path.relative_to(ROOT)))
        old_name = 'co' + 'mem'
        if old_name in path.relative_to(ROOT).as_posix().lower() or old_name in data.decode('utf-8-sig').lower():
            errors.append('Legacy project spelling: '+str(path.relative_to(ROOT)))
    plan=json.loads((ROOT/'runtime/hot_buffer/plan.example.json').read_text())
    assert plan['task_concurrency']==plan['decode_batch_size']==24
    assert plan['hot_chunks']==24 and len(plan['tasks'])==len(set(plan['tasks']))==32
    assert plan['lora_execution']=='peft_merged_bf16'
    assert plan['task_walltime_seconds'] is None and plan['request_timeout'] is None
    assert 'Under review at ICLR 2027.' in (ROOT/'README.md').read_text()
    report=dict(passed=not errors,inventoried_files=len(checked),primary_python_files=python_count,
                hot24_tasks=len(plan['tasks']),errors=errors,
                scope='Static publication validation only; no model inference or benchmark replay.')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if not errors else 1)

if __name__=='__main__':
    main()
