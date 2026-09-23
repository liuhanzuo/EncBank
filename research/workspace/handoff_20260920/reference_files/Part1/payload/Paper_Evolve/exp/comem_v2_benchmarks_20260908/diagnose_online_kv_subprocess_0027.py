"""Reproduce the optional Gaudi probe's stderr encoding, with no model imports."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
command = "pip list | grep habana-torch-plugin"
executed = command
run = subprocess.run(executed, shell=True, capture_output=True, timeout=30,
                     creationflags=subprocess.CREATE_NO_WINDOW)
try:
    run.stderr.decode("utf-8")
    decode_error = None
except UnicodeDecodeError as exc:
    decode_error = {"message": str(exc), "start": exc.start, "byte": exc.object[exc.start]}
result = {
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "command": command,
    "executed_command": executed,
    "console_mode": "CREATE_NO_WINDOW reproduces the hidden queue worker's Chinese Windows shell diagnostics; foreground tool shell emits English.",
    "purpose": "CPU-only byte reproduction of installed bitsandbytes.backends.utils.get_gaudi_sw_version command",
    "torch_imported": False,
    "model_loaded": False,
    "returncode": run.returncode,
    "stdout_bytes": len(run.stdout),
    "stderr_hex": run.stderr.hex(),
    "stderr_cp936": run.stderr.decode("cp936", errors="replace"),
    "utf8_decode_error": decode_error,
    "matches_original_byte_and_position": bool(decode_error and decode_error["start"] == 7 and decode_error["byte"] == 0xb2),
    "production_source": "/srv/encbank/client/AppData/Roaming/Python/Python313/site-packages/bitsandbytes/backends/utils.py:80",
    "original_log_preserved": str(HERE / "results/local/bootstrap_online_kv/logs/pub_lora_0001.log"),
    "impact": "Optional Intel Gaudi software-version probe stderr cannot decode the Chinese Windows missing-grep message as UTF-8. Its stdout is empty, so get_gaudi_sw_version returns None. It does not provide or alter GPU admission, device provenance, or model-process exclusion. No production code was changed and no diagnostic generation rerun.",
}
out = HERE / "logs/online_kv_optional_probe_0027.json"
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
