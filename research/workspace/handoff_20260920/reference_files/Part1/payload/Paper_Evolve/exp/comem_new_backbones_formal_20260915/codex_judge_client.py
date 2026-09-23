"""Use the installed, unmodified official Codex CLI as the requested GPT-6 judge.

The service permits official Codex clients. No HTTP client impersonation, custom
originator headers, or access-control bypass. Per-invocation provider settings
do not edit the user's global Codex configuration. Only classification is allowed.
"""
import json, os, re, subprocess, time
from pathlib import Path
from probe_judge_service import ROOT, conversation_key

MODEL='gpt-6-astra'
BASE='https://sbtunnel.xiaoaojianghu.fun/v1'
EXE=Path('/srv/encbank/client/AppData/Roaming/npm/node_modules/@openai/codex/node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/bin/codex.exe')
WORK=Path('/srv/encbank/client/AppData/Local/MidCacheJudge/empty-workspace')

def call(prompt, output_dir, timeout=120):
    assert EXE.is_file()
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True)
    assert not any(WORK.iterdir()), 'Judge workspace must remain empty'
    key=conversation_key()
    env=os.environ.copy();env['MIDCACHE_JUDGE_API_KEY']=key
    args=[str(EXE),'exec','--ignore-user-config','--skip-git-repo-check','--ephemeral',
          '--sandbox','read-only','--json','--color','never','--model',MODEL,'--cd',str(WORK),
          '--output-last-message',str(out/'answer.txt')]
    config={
        'model_provider':'midcache_sbtunnel',
        'model_providers.midcache_sbtunnel.name':'MidCache authorized judge service',
        'model_providers.midcache_sbtunnel.base_url':BASE,
        'model_providers.midcache_sbtunnel.env_key':'MIDCACHE_JUDGE_API_KEY',
        'model_providers.midcache_sbtunnel.wire_api':'responses',
        'model_providers.midcache_sbtunnel.request_max_retries':0,
        'model_providers.midcache_sbtunnel.stream_max_retries':0,
        'model_providers.midcache_sbtunnel.stream_idle_timeout_ms':60000,
        'model_reasoning_effort':'low','model_reasoning_summary':'none',
        'approval_policy':'never','project_doc_max_bytes':0,'project_root_markers':[],
        'web_search':'disabled',
    }
    for k,v in config.items():args+=['-c',k+'='+json.dumps(v)]
    for feature in ('shell_tool','unified_exec','multi_agent','apps','plugins','browser_use',
                    'browser_use_external','computer_use','image_generation','view_image',
                    'code_mode_host','sleep_tool','workspace_dependencies','skill_search','memories'):
        args+=['--disable',feature]
    args+=['-']
    start=time.monotonic()
    try:
        r=subprocess.run(args,input=prompt,text=True,encoding='utf-8',capture_output=True,env=env,
                         cwd=WORK,timeout=timeout,creationflags=subprocess.CREATE_NO_WINDOW)
        stdout,stderr,code=r.stdout,r.stderr,r.returncode
    except subprocess.TimeoutExpired as e:
        stdout=e.stdout or '';stderr=e.stderr or '';code=None
        if isinstance(stdout,bytes):stdout=stdout.decode('utf-8','replace')
        if isinstance(stderr,bytes):stderr=stderr.decode('utf-8','replace')
    def scrub(s):return re.sub(r'sk-[A-Za-z0-9_-]{24,}','[REDACTED]',s.replace(key,'[REDACTED]'))
    stdout,stderr=scrub(stdout),scrub(stderr)
    (out/'events.jsonl').write_text(stdout,encoding='utf-8')
    (out/'stderr.txt').write_text(stderr,encoding='utf-8')
    answer=scrub((out/'answer.txt').read_text(encoding='utf-8')) if (out/'answer.txt').exists() else ''
    events=[]
    for line in stdout.splitlines():
        try:events.append(json.loads(line))
        except ValueError:pass
    bad=[e for e in events if e.get('item',{}).get('type') in
         ('command_execution','file_change','mcp_tool_call','web_search')]
    result=dict(model_requested=MODEL,base_url=BASE,client='official codex-cli 0.153.4',
        returncode=code,elapsed_s=time.monotonic()-start,answer=answer,tool_actions=len(bad),
        turn_completed=any(e.get('type')=='turn.completed' for e in events),
        usage=[e.get('usage') for e in events if e.get('type')=='turn.completed'],
        errors=[e for e in events if e.get('type') in ('error','turn.failed')],
        credential_persisted=False)
    result['ok']=code==0 and bool(answer.strip()) and result['turn_completed'] and not bad
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    result=call('This is a text-only classification connectivity check. Do not use tools. Reply with exactly CORRECT.',
                ROOT/'judge_gpt6_astra'/'connection')
    print(json.dumps(result,ensure_ascii=True))
