"""Check native Harbor construction, non-thinking RPC, cancellation and history layout."""
import asyncio,json,tempfile,tomllib,subprocess,importlib.metadata
from pathlib import Path
from harbor.models.job.config import JobConfig
import tb_agent
from io_utils import save
from layout import partition,validate_append,fill_recent
ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'plan.json').read_text())
ids=list(range(30000));anchor=ids[:1500]
chunks,q=partition(ids,anchor)
assert anchor+sum(chunks,[])+q[len(anchor):]==ids and 4096<=len(q)-len(anchor)<4608
new_chunks,_=partition(ids+list(range(30000,31000)),anchor);validate_append(chunks,new_chunks)
try:validate_append(chunks,[[999]*512]+chunks[1:]);raise RuntimeError('Stale reuse not rejected')
except AssertionError:pass
assert fill_recent([0],10,4)==([0,7,8,9],[7,8,9])
for arm in P['arms']:
    c=JobConfig.model_validate_json((ROOT/'harbor_configs'/(arm+'.json')).read_text());assert len(c.tasks)==len(P['tasks'])
    agent=tb_agent.MemoryTerminus(logs_dir=ROOT/'cpu_only'/('cancel-async-tasks__'+arm)/'agent',model_name='Qwen3-8B/'+arm,
        **json.loads((ROOT/'harbor_configs'/(arm+'.json')).read_text())['agents'][0]['kwargs'])
    assert agent._llm.arm==arm
for row in P['timeout_inventory']:
    t=tomllib.loads((Path(P['tasks_root'])/row['task']/'task.toml').read_text())
    assert t['agent']['timeout_sec']==row['agent_seconds'] and t['verifier']['timeout_sec']==row['verifier_seconds']
    assert not (Path(P['tasks_root'])/row['task']/'solution').exists()
    subprocess.run(['docker','image','inspect',row['image']],check=True,stdout=subprocess.DEVNULL,timeout=20)
async def rpc():
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        tb_agent.RPC=Path(tmp)
        llm=tb_agent.MemoryLLM('cancel-async-tasks__cpu','iter_k48')
        future=asyncio.create_task(llm.call('test prompt'));await asyncio.sleep(.2)
        req=next(Path(tmp).glob('*.request.json'));d=json.loads(req.read_text())
        text='{"analysis":"Ready","plan":"Inspect","commands":[],"task_complete":true}'
        save(Path(tmp)/(d['request_id']+'.response.json'),dict(status='ok',text=text,prompt_tokens=2,logical_prompt_ids=[1,2],generated_ids=[3]))
        reply=await future;assert reply.content==text and reply.reasoning_content is None
        future=asyncio.create_task(llm.call('next',message_history=[dict(role='user',content='test prompt'),dict(role='assistant',content=text)]))
        await asyncio.sleep(.2);future.cancel();await asyncio.gather(future,return_exceptions=True)
        assert len(list(Path(tmp).glob('*.cancel.json')))==1
asyncio.run(rpc())
save(ROOT/'cpu_check.json',dict(passed=True,harbor=importlib.metadata.version('harbor'),tasks=len(P['tasks']),arms=len(P['arms']),
    docker_images_present=True,nonthinking_answer_retained=True,cancellation_isolated=True,history_partition_lossless=True,stale_archive_rejected=True,official_timeouts=True,model_calls=0))
print((ROOT/'cpu_check.json').read_text())
