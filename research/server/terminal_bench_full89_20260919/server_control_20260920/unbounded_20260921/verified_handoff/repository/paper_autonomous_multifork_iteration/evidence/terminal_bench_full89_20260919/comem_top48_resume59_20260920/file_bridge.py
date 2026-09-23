"""JSON over the previously authorized SSH route; no per-request shared-disk writes."""
from pathlib import Path
import json,sys
from live_mailbox import client
H=Path(__file__).resolve().parent;H.resolve().relative_to(Path('/srv/encbank').resolve());R=H/'run_comem'
action,arm=sys.argv[1:3];assert arm=='comem';ep=R/'transport_endpoint.json'
status={n:json.loads((R/n).read_text()) for n in ['worker_ready.json','worker_failure.json','process_receipt.json','memory_cap_failure.json','worker_complete.json'] if (R/n).exists()}
if action=='status' and (not ep.exists() or any(n in status for n in ['worker_failure.json','process_receipt.json'])):print(json.dumps(status))
else:
 q=dict(action=action)
 if len(sys.argv)>3:q['id']=sys.argv[3]
 if action=='publish':q['request']=json.load(sys.stdin)
 if action=='status':q['event_cursor']=int(sys.argv[3]) if len(sys.argv)>3 else 0
 print(json.dumps(client(json.loads(ep.read_text()),q,timeout=12200 if action=='wait' else 30)))
