import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/encbank_cacheblend_lora_20260916'
r=subprocess.run(['ssh','gpu-node1',f'/srv/encbank/Paper_Evolve/.venv/bin/python -B {REMOTE}/status_remote.py'],capture_output=True,text=True,encoding='utf-8',check=True)
status=json.loads(r.stdout)
(ROOT/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
(ROOT/'launch.json').write_text(json.dumps(status['launch'],indent=2)+'\n')
print(json.dumps(status,indent=2))
