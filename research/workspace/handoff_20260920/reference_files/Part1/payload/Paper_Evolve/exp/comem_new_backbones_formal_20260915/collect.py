import json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
remote='/srv/encbank/comem_new_backbones_formal_20260915'
r=subprocess.run(['ssh','gpu-node1',f'/srv/encbank/Paper_Evolve/.venv/bin/python -B {remote}/status_remote.py'],
                 capture_output=True,text=True,encoding='utf-8',check=True)
status=json.loads(r.stdout)
(root/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
(root/'launch.json').write_text(json.dumps(status['launch'],indent=2)+'\n',encoding='utf-8')
print(json.dumps(status,indent=2))
