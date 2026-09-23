"""One useful, previously unattempted judgment after a read-only service check.

No automatic retry or worker restart; a valid decision is cached for the formal
runner. Credentials travel only over SSH stdin and stay in process memory.
"""
import argparse,json,shlex,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'comem_new_backbones_formal_20260915'))
from probe_judge_service import conversation_key

REMOTE_CODE=r'''
import datetime,json,os,sys,urllib.request,urllib.error
from pathlib import Path
p=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
old=Path('/srv/encbank/comem_new_backbones_formal_20260915')
payload=json.load(sys.stdin);key=payload.pop('key')
os.environ['MIDCACHE_JUDGE_API_KEY']=key
tmp=p/'cache/judge-tmp';tmp.mkdir(parents=True,exist_ok=True);os.environ['TMPDIR']=str(tmp)
sys.path.insert(0,str(old))
from astra_judge_batch import identity,stimulus,write_json
from locomo_judge_reference import PROMPT
from codex_judge_linux import call
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=p/'judge_gpt6_astra';folder=out/'service_checks'/stamp;folder.mkdir(parents=True)
report=dict(at=stamp,attempts=0,credentials_persisted=False,
            direct_codex_user_requested=bool(payload.get('direct_codex')))
try:
    req=urllib.request.Request('https://sbtunnel.xiaoaojianghu.fun/v1/models',headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=20) as response:
        body=json.load(response);report['models_http_status']=response.status
        report['astra_listed']=any(m.get('id')=='gpt-6-astra' for m in body.get('data',[]))
except urllib.error.HTTPError as exc:
    report['models_http_status']=exc.code
except Exception as exc:
    report['models_error_type']=type(exc).__name__
if payload.get('direct_codex') or (report.get('models_http_status')==200 and report.get('astra_listed')):
    protocol=json.loads((old/'judge_protocol.json').read_text())
    selected=None
    for file in sorted((out/'inputs').glob('*.jsonl')):
        for line in file.open(encoding='utf-8'):
            r=json.loads(line)
            if r['status']!='ok' or int(r['category'])==5:continue
            digest=identity(r,protocol);entry=out/'cache'/digest
            if (entry/'decision.json').exists():continue
            if (old/'judge_gpt6_astra/formal_new_models/cache'/digest/'decision.json').exists():continue
            if any(entry.glob('attempt-*')) or any(entry.glob('recovery-probe-*')):continue
            selected=(r,digest,entry);break
        if selected:break
    if selected:
        r,digest,entry=selected
        result=call(PROMPT.format(**stimulus(r)),entry/('recovery-probe-'+stamp),timeout=180)
        label=result['answer'].strip()
        success=result['ok'] and label in ('CORRECT','WRONG')
        report.update(attempts=1,stimulus_digest=digest,judgment_ok=success,returncode=result['returncode'],elapsed_s=result['elapsed_s'],errors=result.get('errors',[]))
        if success:
            write_json(entry/'decision.json',dict(judge_correct=int(label=='CORRECT'),judge_raw=label,usage=result.get('usage'),
                       model='gpt-6-astra',protocol_id=protocol['protocol_id'],stimulus_digest=digest))
            report['decision_saved_for_formal_reuse']=True
    else:report['no_unattempted_pending_stimulus']=True
write_json(folder/'report.json',report)
write_json(out/'latest_service_check.json',dict(report_path=str(folder/'report.json'),**report))
print(json.dumps(report))
'''

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--direct-codex',action='store_true',
        help='One explicitly requested actual Codex judgment even if model-list access is denied.')
    args=parser.parse_args()
    key=conversation_key()
    command='/srv/encbank/Paper_Evolve/.venv/bin/python -B -c '+shlex.quote(REMOTE_CODE)
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-node1',command],
                     input=json.dumps(dict(key=key,direct_codex=args.direct_codex)),text=True,capture_output=True,timeout=230)
    if r.returncode:raise RuntimeError(r.stderr.replace(key,'[REDACTED]')[:700])
    report=json.loads(r.stdout)
    (ROOT/'latest_service_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
