import asyncio,json,subprocess,tempfile,tomllib,importlib.metadata
from pathlib import Path
from harbor.models.job.config import JobConfig
import tb_agent
from io_utils import save
ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'plan.json').read_text())
for arm in P['arms']:
    c=JobConfig.model_validate_json((ROOT/'harbor_configs'/(arm+'.json')).read_text());assert len(c.tasks)==6
    agent=tb_agent.MemoryTerminus(logs_dir=ROOT/'cpu_only'/('cancel-async-tasks__'+arm)/'agent',model_name='Qwen3.8-27B/'+arm,
        **json.loads((ROOT/'harbor_configs'/(arm+'.json')).read_text())['agents'][0]['kwargs'])
    assert agent._llm.arm==arm and agent._llm.get_model_context_limit()==262144 and agent._llm.get_model_output_limit()==32768
for row in P['timeout_inventory']:
    task=tomllib.loads((Path(P['tasks_root'])/row['task']/'task.toml').read_text())
    assert task['agent']['timeout_sec']==row['agent_seconds'] and task['verifier']['timeout_sec']==row['verifier_seconds']
    subprocess.run(['docker','image','inspect',task['environment']['docker_image']],stdout=subprocess.DEVNULL,check=True,timeout=20)
async def check():
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        tb_agent.RPC=Path(d);a=tb_agent.MemoryLLM('cancel-async-tasks__probe','iter_k48')
        f=asyncio.create_task(a.call('Inspect terminal'));await asyncio.sleep(.2)
        req=next(Path(d).glob('*.request.json'));q=json.loads(req.read_text())
        content='{"analysis":"Ready","plan":"Inspect","commands":[],"task_complete":true}'
        save(Path(d)/(q['request_id']+'.response.json'),dict(status='ok',text='thinking</think>'+content,prompt_tokens=2,prompt_ids=[1,2],generated_ids=[3]))
        reply=await f;assert reply.content==content and reply.reasoning_content=='thinking'
asyncio.run(check())
save(ROOT/'cpu_check.json',dict(passed=True,harbor=importlib.metadata.version('harbor'),tasks=6,arms=4,original_timeouts=True,
    thinking_content_preserved=True,model_calls=0,gpu_probe_check_pending=True))
print((ROOT/'cpu_check.json').read_text())
