"""Official Terminus2 agent with a native, remote GPU model transport only."""
import asyncio,json,os,time,uuid
from pathlib import Path
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM,LLMResponse,ContextLengthExceededError,OutputLengthExceededError
from harbor.models.metric import UsageInfo
REMOTE='/srv/encbank/qencbank_align_codex_20260911/terminal_bench_20260918'
PY='/srv/encbank/qencbank_runtime_20260911/python312/bin/python'
SSH='/mnt/c/Windows/System32/OpenSSH/ssh.exe'
async def exchange(arm,data):
    p=await asyncio.create_subprocess_exec(SSH,'-o','BatchMode=yes','-o','ConnectTimeout=15','gpu-node1',f'{PY} {REMOTE}/bridge.py {arm}',stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    try:
        out,err=await p.communicate((json.dumps(data)+'\n').encode())
        if p.returncode:raise RuntimeError(f'SSH mailbox exit {p.returncode}: '+err.decode(errors='replace')[-4000:])
        return json.loads(out)
    finally:
        if p.returncode is None:p.kill();await p.wait()
class MailboxLLM(BaseLLM):
    def __init__(self,arm):self.arm=arm;self.task_id='tb_'+uuid.uuid4().hex;self.step=0;self.start=None;self.last_messages=[]
    def get_model_context_limit(self):return 36864
    def get_model_output_limit(self):return 4096
    async def call(self,prompt,**kwargs):
        history=kwargs.get('message_history') or [];messages=history+[dict(role='user',content=prompt)]
        assert len(history)==2*self.step, 'Unexpected summary/branching of registered append-only trajectory'
        if self.last_messages:assert history[:-1]==self.last_messages, 'Prior chat changed'
        self.last_messages=messages
        if self.start is None:self.start=time.monotonic()
        rid=f'{self.task_id}_{self.step:05d}'
        request=dict(action='call',request_id=rid,task_id=self.task_id,step=self.step,messages=messages,remaining_seconds=max(0,900-(time.monotonic()-self.start)))
        began=time.monotonic()
        try:r=await exchange(self.arm,request)
        except asyncio.CancelledError:
            await asyncio.shield(exchange(self.arm,dict(action='cancel',request_id=rid)))
            raise
        r['transport_roundtrip_seconds']=time.monotonic()-began
        log=kwargs.get('logging_path')
        if log:
            p=Path(log);p.parent.mkdir(parents=True,exist_ok=True);p.with_suffix('.native.json').write_text(json.dumps(r,indent=2)+'\n')
        if r['status']=='context_limit':raise ContextLengthExceededError('Registered native context limit')
        if r['status']=='deadline':raise TimeoutError('Registered per-task agent deadline')
        text=r['text'];reasoning=None
        if '</think>' in text:reasoning,text=text.split('</think>',1)
        else:reasoning,text=text,''  # Never execute an unclosed thinking segment as terminal commands.
        self.step+=1
        return LLMResponse(content=text.strip(),reasoning_content=reasoning,model_name='Qwen3.8-27B/'+self.arm,usage=UsageInfo(prompt_tokens=r['prompt_tokens'],completion_tokens=r['generated_tokens'],cache_tokens=min(r['prompt_tokens'],r.get('cached_prefix_tokens',0)),cost_usd=0),completion_token_ids=r['generated_ids'],extra={k:v for k,v in r.items() if k not in ['text','generated_ids']})
class QEncbankTerminus(Terminus2):
    def _init_llm(self,**kwargs):
        arm=kwargs['model_name'].split('/')[-1];assert arm in ['dense','raw_shared','encbank'];return MailboxLLM(arm)
