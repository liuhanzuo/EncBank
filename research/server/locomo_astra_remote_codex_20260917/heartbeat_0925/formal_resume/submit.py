"""Submit once; inherit no shared account credentials and persist no secret."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
H=Path(__file__).resolve().parent
def main():
    root=Path('/srv/encbank').resolve();assert H.is_relative_to(root)
    expected=sys.argv[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(H/'plan.json')==expected
    m=json.loads((H/'manifest.json').read_text());assert all(sha(H/n)==s for n,s in m['files'].items())
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf-8','replace')
            if (p/'exe').resolve().name=='codex' and ' exec ' in cmd and ('qcomem_astra' in cmd or 'locomo_astra_remote_codex' in cmd):active.append(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert not active,'Existing remote judge is active'
    (H/'submission_once.lock').mkdir(exist_ok=False)
    payload=json.loads(sys.stdin.buffer.read());assert payload['key']
    env=os.environ.copy()
    for name in ('CODEX_API_KEY','CODEX_ACCESS_TOKEN','OPENAI_API_KEY','QCOMEM_JUDGE_API_KEY'):env.pop(name,None)
    argv=[sys.executable,'-X','utf8','-B',str(H/'coordinate.py'),expected]
    with (H/'coordinator_stdout.log').open('xb') as out,(H/'coordinator_stderr.log').open('xb') as err:
        child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=out,stderr=err,cwd=H,env=env,start_new_session=True)
        child.stdin.write(json.dumps(payload).encode());child.stdin.close()
    start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    receipt={'status':'SUBMITTED_NOT_COMPLETE','pid':child.pid,'proc_start_ticks':start_ticks,'argv':argv,
             'plan_sha256':expected,'at':datetime.datetime.now().astimezone().isoformat(),
             'no_active_remote_judge_before_launch':True,'credential_in_argv_or_file':False,'GPU_used':False}
    (H/'submission_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
