import json, os
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = [
    ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
]
root = Path(__file__).resolve().parent
index = int(os.environ["SLURM_ARRAY_TASK_ID"])
repo, revision = MODELS[index]
destination = root / "models" / repo.split("/")[1]
print(json.dumps({"status": "downloading", "repo": repo, "revision": revision}), flush=True)
snapshot_download(repo, revision=revision, local_dir=destination,
                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken"],
                  max_workers=4)
config = json.loads((destination / "config.json").read_text())
record = {"repo": repo, "revision": revision, "path": str(destination),
          "model_type": config["model_type"], "text_config": config.get("text_config")}
(destination / "download_complete.json").write_text(json.dumps(record, indent=2))
print(json.dumps({"status": "ready", **record}), flush=True)
