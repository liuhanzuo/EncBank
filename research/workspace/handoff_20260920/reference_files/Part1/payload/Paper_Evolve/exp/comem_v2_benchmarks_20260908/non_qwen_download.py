"""Anonymous fixed-revision, whitelisted SmolLM2 download; no model import."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import traceback
from urllib.request import Request, urlopen

REPO = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
REVISION = "31b70e2e869a7173562077fd711b654946d38674"
ALLOWED_ROOT = Path("/data/liuhanzuo/comem_v2_20260908/models/SmolLM2-1.7B-Instruct--31b70e2e869a")
FILES = ["config.json", "generation_config.json", "tokenizer_config.json",
         "special_tokens_map.json", "tokenizer.json", "vocab.json", "merges.txt", "README.md", "model.safetensors"]

def now():
    return datetime.now(timezone.utc).isoformat()

def write(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temporary.replace(path)

def request(url):
    return urlopen(Request(url, headers={"User-Agent": "CoMem-non-Qwen-fixed-revision-preparation/1"}), timeout=60)

def verify(path, meta):
    size = path.stat().st_size
    sha256, blob = hashlib.sha256(), hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            sha256.update(block)
            blob.update(block)
    assert size == meta["size"], (path.name, size, meta["size"])
    if meta.get("lfs"):
        assert sha256.hexdigest() == meta["lfs"]["sha256"], (path.name, "LFS SHA256")
    else:
        assert blob.hexdigest() == meta["blobId"], (path.name, "Git blob identity")
    return {"name": path.name, "bytes": size, "sha256": sha256.hexdigest(),
            "official_lfs_sha256": meta.get("lfs", {}).get("sha256"),
            "official_blob_id": meta["blobId"], "integrity_verified": True}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ALLOWED_ROOT)
    ap.add_argument("--attempt", type=int, required=True)
    args = ap.parse_args()
    assert args.model_dir.resolve() == ALLOWED_ROOT
    assert args.attempt > 0
    args.model_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.model_dir / f"non_qwen_download_attempt{args.attempt}.json"
    assert not receipt.exists(), "retain all prior attempts"
    api = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    state = {"status": "starting", "started_utc": now(), "repo": REPO, "revision": REVISION,
             "model_dir": str(args.model_dir), "source_url": api, "authenticated": False,
             "additional_terms_accepted": False, "model_loaded": False, "files": []}
    write(receipt, state)
    try:
        with request(api) as response:
            source = json.load(response)
        assert source["sha"] == REVISION and not source["private"] and source["gated"] is False
        assert source.get("cardData", {}).get("license") == "apache-2.0"
        metadata = {x["rfilename"]: x for x in source["siblings"]}
        assert all(n in metadata for n in FILES)
        write(args.model_dir / "non_qwen_official_source.json", source)
        required = sum(metadata[n]["size"] for n in FILES)
        free = shutil.disk_usage(args.model_dir).free
        assert free > required * 2 + 1024 ** 3, (free, required)
        state.update(status="downloading", available_bytes_before=free, required_file_bytes=required,
                     whitelist=FILES, private=False, gated=False, license="apache-2.0")
        write(receipt, state)
        for name in FILES:
            path = args.model_dir / name
            state.update(active_file=name, updated_utc=now())
            write(receipt, state)
            if path.exists():
                proof = verify(path, metadata[name])
                proof["reused_verified_file"] = True
            else:
                partial = args.model_dir / f"{name}.partial.attempt{args.attempt}"
                assert not partial.exists()
                url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
                with request(url) as response, partial.open("xb") as stream:
                    state["active_final_url_host"] = response.url.split("/")[2]
                    count = 0
                    while block := response.read(4 * 1024 * 1024):
                        stream.write(block)
                        count += len(block)
                        state.update(active_file_bytes=count, updated_utc=now())
                        write(receipt, state)
                proof = verify(partial, metadata[name])
                partial.rename(path)
                proof.update(name=name, source_url=url, reused_verified_file=False)
            state["files"].append(proof)
            write(receipt, state)
            print(json.dumps({"file": name, "bytes": proof["bytes"], "verified": True}), flush=True)
            if name == "README.md":
                write(args.model_dir / "non_qwen_TOKENIZER_READY.json", {**state, "status": "tokenizer_ready"})
        state.update(status="complete", completed_utc=now(), active_file=None, active_file_bytes=None)
        write(receipt, state)
        write(args.model_dir / "non_qwen_DOWNLOAD_READY.json", state)
        return 0
    except Exception:
        state.update(status="failed", failed_utc=now(), error=traceback.format_exc())
        write(receipt, state)
        print(state["error"], flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
