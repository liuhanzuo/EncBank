"""Server-only opaque task copy; only task.toml is parsed, no verifier or answer inspection."""
import hashlib
import json
import os
import shutil
import time
import tomllib
from pathlib import Path, PurePosixPath

BASE = Path('/srv/encbank/COMem_Migration_20260920/Part2/payload/qcomem/.runtime/terminal_bench_full89_20260919')
RUNTIME = Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920')
H = Path(__file__).resolve().parent


def main():
    assert os.name == 'posix'
    RUNTIME.resolve().relative_to(Path('/srv/encbank').resolve())
    source_manifest = json.loads((H / 'runs/comem_k12_server_r6_20260920/task_manifest.json').read_text())
    locations = json.loads((RUNTIME / 'task_source_locations.json').read_text())
    migration = Path('/srv/encbank/COMem_Migration_20260920')
    configurations = {}
    prefix = 'Part2/payload/qcomem/.runtime/terminal_bench_full89_20260919/tasks_no_total_deadline_20260920/'
    with (migration / '_transfer/UPLOAD_MANIFEST.jsonl').open() as upload_manifest:
        for line in upload_manifest:
            item = json.loads(line)
            if item['path'].startswith(prefix) and item['path'].endswith('/task.toml'):
                configurations[item['path'][len(prefix):]] = item
    destination = RUNTIME / 'tasks'
    rows = []
    for row in source_manifest['files']:
        relative = PurePosixPath(row['path'])
        assert relative.parts[0] == 'tasks' and '..' not in relative.parts
        relative = Path(*relative.parts[1:])
        original = BASE / 'tasks' / relative
        copied_source = BASE / 'tasks_no_total_deadline_20260920' / relative
        if relative.name == 'task.toml':
            data = copied_source.read_bytes()
            expected = configurations[relative.as_posix()]
            assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256']
            after = tomllib.loads(data.decode())
            assert after.get('agent', {}).get('timeout_sec') is None
            if original.exists():
                original_data = original.read_bytes()
                assert hashlib.sha256(original_data).hexdigest() == row['sha256'], relative
                before = tomllib.loads(original_data.decode())
                before.get('agent', {}).pop('timeout_sec', None)
                assert after == before, 'Unexpected task protocol modification: ' + str(relative)
        else:
            # Upload deduplication can leave an alias absent until all shards finish.
            # Reuse only an already-present file with exactly the frozen content hash.
            if row['bytes'] == 0:
                data = b''
            else:
                location = migration / locations[row['sha256']]
                location.resolve().relative_to(migration.resolve())
                data = location.read_bytes()
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], relative
        target = destination / relative
        target.resolve().relative_to(RUNTIME.resolve())
        if target.exists():
            assert target.read_bytes() == data, 'Refuse to overwrite modified task: ' + str(relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        rows.append({'path': relative.as_posix(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                     'opaque_verifier': row.get('opaque_verifier', False)})
    report = {'status': 'PASS', 'epoch': time.time(), 'revision': source_manifest['revision'],
              'files': rows, 'task_count': len(source_manifest['tasks']),
              'verifier_contents_inspected': False, 'task_protocol_change': 'None relative to the frozen no-deadline predecessor; task.toml verified against its migration SHA, other files against the original task manifest'}
    (RUNTIME / 'task_copy_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'tasks': report['task_count'], 'files': len(rows)}))


if __name__ == '__main__':
    main()
