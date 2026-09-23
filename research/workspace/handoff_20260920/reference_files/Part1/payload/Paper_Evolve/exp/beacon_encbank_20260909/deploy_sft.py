"""Upload this isolated experiment; launch its controller only with --start."""
import argparse
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
ROOT = '/data/liuhanzuo/encbank_v2_20260908'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'longjing-1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    bundle = HERE / 'sft_payload.tar.gz'
    files = list(HERE.glob('*.py')) + list(HERE.glob('*.md'))
    files += list((HERE / 'data/qasper_sft').glob('*.json*'))
    with tarfile.open(bundle, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(WORK).as_posix())
    remote_bundle = ROOT + '/beacon_sft_payload.tar.gz'
    subprocess.run(['scp', '-o', 'BatchMode=yes', str(bundle),
                    'longjing-1:' + remote_bundle], check=True, timeout=180)
    subprocess.run(SSH + ['tar -xzf ' + shlex.quote(remote_bundle) + ' -C ' + shlex.quote(ROOT + '/workspace')],
                   check=True, timeout=60)
    print(f'Uploaded {len(files)} files ({bundle.stat().st_size} compressed bytes)', flush=True)
    if args.start:
        source = ("import pathlib,subprocess; "
                  f"root=pathlib.Path({ROOT!r}); "
                  "out=root/'outputs/beacon_sft_20260909'; out.mkdir(exist_ok=True); "
                  "log=(out/'controller.log').open('a'); "
                  "p=subprocess.Popen([str(root/'venv/bin/python'),'-u',"
                  "str(root/'workspace/exp/beacon_encbank_20260909/remote_sft_queue.py')],"
                  "cwd=root/'workspace',stdout=log,stderr=subprocess.STDOUT,start_new_session=True); "
                  "print(p.pid)")
        subprocess.run(SSH + [shlex.quote(ROOT + '/venv/bin/python') + ' -c ' + shlex.quote(source)],
                       check=True, timeout=30)


if __name__ == '__main__':
    main()
