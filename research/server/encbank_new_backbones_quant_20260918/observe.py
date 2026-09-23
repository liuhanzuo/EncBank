"""Bounded read-only observation and local collection of small status artifacts."""
import json, shlex, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918'

def main():
    script=r'''
import json,os
from pathlib import Path
p=Path('/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918')
files=['status.json','preflight_progress.json','preflight_parent_exit.json','coordinator_failure.json','coordinator_launch.json','judge_gpt6_astra/watch_status.json','judge_gpt6_astra/watch_failure.json','judge_gpt6_astra/launch.json']
out={n:json.loads((p/n).read_text()) for n in files if (p/n).exists()}
for name in ('coordinator_launch.json','judge_gpt6_astra/launch.json'):
    if name in out:
        cmd=Path('/proc')/str(out[name]['pid'])/'cmdline'
        out[name]['process_alive']=cmd.exists()
print(json.dumps(out))
'''
    cmd='/srv/encbank/Paper_Evolve/.venv/bin/python -c '+shlex.quote(script)
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-node1',cmd],text=True,capture_output=True,timeout=35)
    assert r.returncode==0,r.stderr
    data=json.loads(r.stdout);(ROOT/'observed_status.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    s=data.get('status.json',{})
    print(json.dumps(dict(phase=s.get('phase'),completed=s.get('completed'),failed=s.get('failed'),
        gpu_requests=s.get('total_owned_gpu_requests'),queue=s.get('owned_queue'),
        preflight=data.get('preflight_progress.json'),preflight_exit=data.get('preflight_parent_exit.json'),
        coordinator=data.get('coordinator_launch.json'),judge=data.get('judge_gpt6_astra/watch_status.json'),
        error=data.get('coordinator_failure.json'),judge_error=data.get('judge_gpt6_astra/watch_failure.json'))))

if __name__=='__main__':main()
