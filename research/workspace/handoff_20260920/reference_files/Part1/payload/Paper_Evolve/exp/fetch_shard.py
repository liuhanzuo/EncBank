"""Resumable single-file fetch from the HF CDN.

huggingface_hub's downloader was restarting shard 1 from byte 0 on every retry --
three partial copies of the same blob, 6.6 GB of wasted transfer -- because each
attempt opens a new `.<hash>.incomplete` rather than continuing the previous one.
Over a slow, flaky proxy that never converges.  This does the obvious thing instead:
one file, HTTP Range resume, verify sha256 against the blob name, then place it.
"""

import hashlib
import shutil
import sys
import time
import urllib.request
from pathlib import Path

REPO = "Qwen/Qwen3-1.7B"
HUB = Path.home() / ".cache/huggingface/hub" / ("models--" + REPO.replace("/", "--"))
CHUNK = 1 << 20


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "python-requests/2")
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status in (301, 302, 307, 308):
            return head_size(r.headers["Location"])
        return int(r.headers["Content-Length"])


def fetch(url: str, dst: Path, total: int, tries: int = 200) -> None:
    for attempt in range(tries):
        have = dst.stat().st_size if dst.exists() else 0
        if have >= total:
            return
        req = urllib.request.Request(url)
        req.add_header("Range", f"bytes={have}-")
        req.add_header("User-Agent", "python-requests/2")
        try:
            t0, start = time.time(), have
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "ab") as f:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if have % (64 << 20) < CHUNK:
                        el = max(time.time() - t0, 1e-6)
                        print(
                            f"  {have/1e9:.2f}/{total/1e9:.2f} GB "
                            f"({100*have/total:.1f}%)  {(have-start)/el/1e6:.1f} MB/s",
                            flush=True,
                        )
        except Exception as e:  # slow proxy drops connections constantly; just resume
            print(f"  [retry {attempt}] {type(e).__name__}: {e}", flush=True)
            time.sleep(2)
    raise RuntimeError(f"gave up at {dst.stat().st_size}/{total}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(filename: str, blob_sha: str) -> None:
    url = f"https://huggingface.co/{REPO}/resolve/main/{filename}"
    blobs, snaps = HUB / "blobs", HUB / "snapshots"
    snap = next(snaps.iterdir())

    total = head_size(url)
    part = blobs / f"{blob_sha}.incomplete"
    print(f"{filename}: {total/1e9:.2f} GB, have {part.stat().st_size/1e9 if part.exists() else 0:.2f} GB")

    # drop the restarted-from-zero duplicates, keep the furthest-along partial
    for stale in blobs.glob(f"{blob_sha}.*.incomplete"):
        print(f"  removing stale partial {stale.name} ({stale.stat().st_size/1e9:.2f} GB)")
        stale.unlink()

    fetch(url, part, total)

    print("  verifying sha256 ...", flush=True)
    got = sha256(part)
    if got != blob_sha:
        raise SystemExit(f"SHA MISMATCH\n  want {blob_sha}\n  got  {got}")
    print("  sha ok")

    final = blobs / blob_sha
    part.rename(final)
    # this machine has no symlink privilege, so the cache is in copy mode
    shutil.copy2(final, snap / filename)
    print(f"  placed {snap / filename}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
