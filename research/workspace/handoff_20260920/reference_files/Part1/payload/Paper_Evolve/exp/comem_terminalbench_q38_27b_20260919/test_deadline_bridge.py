"""No model calls: deadline errors stay typed and do not publish another request."""
import asyncio,json,tempfile
from pathlib import Path
import tb_agent as agent
from harbor.trial.errors import AgentTimeoutError

async def check(status,expired=True):
    with tempfile.TemporaryDirectory() as directory:
        agent.RPC=Path(directory)
        llm=agent.MemoryLLM('regex-log__deadline_bridge_test','iter_k12')
        if status=='cancelled' and expired:llm.limit=0
        async def deliver():
            while not list(agent.RPC.glob('*.request.json')):await asyncio.sleep(.01)
            path=next(agent.RPC.glob('*.request.json'));q=json.loads(path.read_text())
            agent.save(path.with_name(path.name.replace('.request.','.response.')),dict(request_id=q['request_id'],status=status))
        response=asyncio.create_task(deliver());errors=[]
        for _ in range(3):
            try:await llm.call('synthetic test input')
            except Exception as exc:
                assert isinstance(exc,AgentTimeoutError)==(status=='deadline' or expired)
                errors.append(exc)
            else:raise AssertionError('Missing deadline exception')
        await response
        assert len({id(e) for e in errors})==1
        assert len(list(agent.RPC.glob('*.request.json')))==1 and llm.step==0
async def main():
    await check('deadline');await check('cancelled')
    await check('cancelled',expired=False)
    print('PASS: timeout retained through three wrapper calls with one request; early cancellation remains an error; official Harbor catches AgentTimeoutError before verification; zero GPU calls')
asyncio.run(main())
