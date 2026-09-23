"""Small stdlib-only deployment helper for the isolated sparse CoMem experiment.

No model is loaded locally. Remote GPU work must enter through remote_sparse_queue.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ROOT = "/data/liuhanzuo/comem_v2_20260908"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "longjing-1"]
CODE = ROOT + "/workspace/exp/comem_block_attention_20260911"


def remote_run(arguments, *, cpu=False, timeout=300):
    env = ["OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2",
           "TOKENIZERS_PARALLELISM=false", "PYTHONUNBUFFERED=1"]
    if cpu:
        env.append("CUDA_VISIBLE_DEVICES=")
    command = ["env"] + env + [ROOT + "/venv/bin/python"] + list(arguments)
    return subprocess.run(SSH + [" ".join(shlex.quote(x) for x in command)],
                          check=True, timeout=timeout)


def deploy():
    payload = HERE / "sparse_payload.tar.gz"
    files = sorted(p for p in HERE.iterdir()
                   if p.is_file() and p.suffix in {".py", ".md", ".json"})
    with tarfile.open(payload, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(WORKSPACE).as_posix(), recursive=False)
    remote_payload = ROOT + "/sparse_payload_20260911.tar.gz"
    subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                    str(payload), "longjing-1:" + remote_payload], check=True, timeout=180)
    # The archive was created above from regular local files within this new task.
    subprocess.run(SSH + ["tar -xzf " + shlex.quote(remote_payload)
                          + " -C " + shlex.quote(ROOT + "/workspace")],
                   check=True, timeout=60)
    print(f"Deployed {len(files)} task files to {CODE}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("deploy", "cpu", "status"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "deploy":
        if args.arguments:
            parser.error("deploy takes no extra arguments")
        deploy()
    elif args.action == "status":
        command = [ROOT + "/venv/bin/python", CODE + "/inspect_remote.py"]
        result = subprocess.run(SSH + [" ".join(shlex.quote(x) for x in command)],
                                check=True, capture_output=True, text=True, timeout=30)
        import json
        data = json.loads(result.stdout)
        destination = HERE / "results"
        destination.mkdir(exist_ok=True)
        (destination / "remote_status.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
    else:
        values = args.arguments
        if values and values[0] == "--":
            values = values[1:]
        if not values:
            parser.error("cpu requires a remote Python script and arguments")
        remote_run(values, cpu=True, timeout=900)


if __name__ == "__main__":
    main()
