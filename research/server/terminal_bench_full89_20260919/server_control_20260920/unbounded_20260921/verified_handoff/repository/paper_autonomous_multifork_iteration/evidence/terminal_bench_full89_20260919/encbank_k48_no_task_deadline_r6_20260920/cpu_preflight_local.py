"""Focused native Harbor, task identity, input hashes and concurrent transport check."""
from pathlib import Path
import asyncio,hashlib,json,tempfile,tomllib
import tb_agent_rpc as rpc
from harbor.models.job.config import JobConfig
H=Path(__file__).resolve().parent
R=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919')
P=json.loads((H/'plan.json').read_text());cfg=JobConfig.model_validate_json((H/'encbank_harbor_template.json').read_text())
manifest=json.loads((H/'task_manifest.json').read_text());assert len(cfg.tasks)==len(P['tasks'])
for row in manifest['files']:
    path=R/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
for name in P['tasks']:
    task=tomllib.loads((Path(P['task_root'])/name/'task.toml').read_text())
    registered=next(x for x in P['timeout_inventory'] if x['task']==name)
    assert task['agent'].get('timeout_sec') is None
    assert task['environment'].get('gpus',0)==0
    assert not (R/'tasks'/name/'solution').exists()
agent=rpc.ConcurrentEncbankTerminus(logs_dir=R/'cpucheck/fix-git__check/agent',model_name='Qwen3.8-27B/dense',**json.loads((H/'encbank_harbor_template.json').read_text())['agents'][0]['kwargs'])
assert agent._llm.task=='fix-git' and agent._llm.limit is None
async def transport_check():
    with tempfile.TemporaryDirectory(dir=R,prefix='cpu_rpc_') as d:
        rpc.RPC=Path(d);a=rpc.MailboxLLM('fix-git__a');b=rpc.MailboxLLM('fix-git__b')
        fa=asyncio.create_task(a.call('inputA'));fb=asyncio.create_task(b.call('inputB'))
        await asyncio.sleep(.2)
        paths=list((Path(d)/'encbank').glob('*.request.json'));assert len(paths)==2 and a.task_id!=b.task_id
        for path in paths:
            q=json.loads(path.read_text());assert q['remaining_seconds']==12000
            reply=dict(status='ok',text='reason</think>'+q['messages'][-1]['content'],prompt_tokens=2,prompt_ids=[1,2],generated_tokens=2,generated_ids=[3,4])
            rpc.save(path.with_name(q['request_id']+'.response.json'),reply)
        ra,rb=await asyncio.gather(fa,fb);assert ra.content=='inputA' and rb.content=='inputB'
        assert a.step==b.step==1 and a.last_messages!=b.last_messages
        # Cancellation must produce a per-request signal without aborting another task.
        fc=asyncio.create_task(rpc.MailboxLLM('fix-git__c').call('inputC'));await asyncio.sleep(.2);fc.cancel()
        await asyncio.gather(fc,return_exceptions=True);assert len(list((Path(d)/'encbank').glob('*.cancel.json')))==1
asyncio.run(transport_check())
out=dict(status='PASS',tasks=len(P['tasks']),files=len(manifest['files']),total_task_timeout_None=True,per_call_transport_watchdog_seconds=12000,model_calls=0,
    native_harbor_constructor=True,concurrent_task_isolation=True,cancellation_isolated=True,solutions_absent=True,
    verifier_contents_inspected=False,plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest())
(H/'cpu_preflight_local.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
