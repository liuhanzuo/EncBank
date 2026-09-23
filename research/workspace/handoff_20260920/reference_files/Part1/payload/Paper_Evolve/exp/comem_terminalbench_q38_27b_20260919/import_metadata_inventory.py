"""Read only RECORD listings, never recurse over package files."""
import os, json, csv, io
from pathlib import Path
roots = ['/srv/encbank/Paper_Evolve/.venv/lib64/python3.12/site-packages',
         '/srv/encbank/comem_infra_recheck_20260912/deps']
rows = []
for root in roots:
    for entry in os.scandir(root):
        if not entry.name.endswith(('.dist-info', '.egg-info')):
            continue
        path = Path(entry.path)
        record = path / 'RECORD'
        try:
            declared = (path / 'top_level.txt').read_text().split()
        except FileNotFoundError:
            declared = []
        try:
            paths = [r[0] for r in csv.reader(io.StringIO(record.read_text()))]
        except FileNotFoundError:
            paths = []
        rows.append(dict(root=root, name=entry.name, declared=declared, files=len(paths),
                         first=paths[:1], last=paths[-1:]))
print(json.dumps(rows))
