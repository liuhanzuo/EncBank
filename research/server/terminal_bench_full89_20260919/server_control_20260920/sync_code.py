"""Upload source only to the unused server staging directory; never launch experiments."""
import subprocess
import sys
import tarfile
from pathlib import Path

H = Path(__file__).resolve().parent
subprocess.run([sys.executable, '-X', 'utf8', str(H / 'build_runs.py')], check=True)
with tarfile.open(H / 'server_control.tar.gz', 'w:gz') as archive:
    for path in H.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.json', '.slurm'} and 'reports' not in path.parts:
            archive.add(path, arcname=path.relative_to(H).as_posix())
subprocess.run(['scp', str(H / 'server_control.tar.gz'), 'gpu-node1:/srv/encbank/qencbank_runtime_20260911/server_control_20260920/server_control.tar.gz'], check=True)
remote = """from pathlib import Path
import tarfile
base=Path('/srv/encbank')
root=base/'qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920'
root.resolve().relative_to(base.resolve())
assert not any(root.glob('runs/*/submission.json')) and not any(root.glob('runs/*/execution/owner_registration.json')),'Refuse to overwrite a used server run'
with tarfile.open(base/'qencbank_runtime_20260911/server_control_20260920/server_control.tar.gz') as archive:
 assert all(member.isfile() for member in archive.getmembers())
 archive.extractall(root,filter='data')
print('Server source synchronized; no jobs submitted')
"""
subprocess.run(['ssh', '-o', 'BatchMode=yes', 'gpu-node1', '/srv/encbank/qencbank_runtime_20260911/python312/bin/python', '-'], input=remote.encode(), check=True)
