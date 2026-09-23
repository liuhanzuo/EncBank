"""Official Terminus2 with isolated requests to one Qwen3-8B service."""
import asyncio,hashlib,json,time
from pathlib import Path
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM,LLMResponse,ContextLengthExceededError
from harbor.models.metric import UsageInfo
from io_utils import save
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'plan.json').read_text())
RPC=ROOT/'rpc'

class MemoryLLM(BaseLLM):
    def __init__(self,trial,arm):
        super().__init__();assert arm in P['arms']
        self.trial=trial;self.arm=arm;self.task=trial.split('__')[0]
        self.task_id=hashlib.sha256((arm+'/'+trial).encode()).hexdigest()[:24]
        self.step=0;self.start=None;self.failed=False;self.last=[]
        self.limit=next(x['agent_seconds'] for x in P['timeout_inventory'] if x['task']==self.task)
    def get_model_context_limit(self):return P['context_tokens']
    def get_model_output_limit(self):return P['max_new_tokens']
    async def call(self,prompt,**kwargs):
        assert not self.failed,'No repeated generation after transport/model failure'
        history=kwargs.get('message_history') or []
        messages=history+[dict(role='user',content=prompt)]
        if self.last:assert history[:len(self.last)]==self.last,'Previous history changed'
        if self.start is None:self.start=time.monotonic()
        began=time.monotonic();rid=f'{self.task_id}_{self.step:05d}'
        req=RPC/(rid+'.request.json');resp=RPC/(rid+'.response.json');err=RPC/(rid+'.error.json')
        assert not req.exists(),'Refuse duplicate request'
        remaining=max(0,self.limit-(began-self.start))
        save(req,dict(request_id=rid,task_id=self.task_id,trial=self.trial,task=self.task,arm=self.arm,
            step=self.step,messages=messages,remaining_seconds=remaining,client_created_epoch=time.time()))
        try:
            while not resp.exists():
                if err.exists():raise RuntimeError(json.loads(err.read_text())['error'])
                if time.monotonic()-began>remaining+30:raise TimeoutError('Agent deadline elapsed')
                await asyncio.sleep(.1)
            r=json.loads(resp.read_text())
        except BaseException:
            self.failed=True;save(RPC/(rid+'.cancel.json'),dict(cancelled=True));raise
        if r['status']=='context_limit':
            self.failed=True;raise ContextLengthExceededError(r['error'])
        if r['status']!='ok':self.failed=True;raise RuntimeError(json.dumps(r))
        self.last=messages;self.step+=1
        r['client_roundtrip_seconds']=time.monotonic()-began
        save(RPC/(rid+'.timing.json'),dict(client_roundtrip_seconds=r['client_roundtrip_seconds']))
        # Non-thinking mode: retain the entire answer, including JSON tool commands.
        return LLMResponse(content=r['text'].strip(),reasoning_content=None,model_name='Qwen3-8B/'+self.arm,
            usage=UsageInfo(prompt_tokens=r['prompt_tokens'],completion_tokens=len(r['generated_ids']),
                cache_tokens=r.get('cached_prefix_tokens',0),cost_usd=0),
            prompt_token_ids=r['logical_prompt_ids'],completion_token_ids=r['generated_ids'],
            extra={k:v for k,v in r.items() if k not in ['text','logical_prompt_ids','generated_ids']})

class MemoryTerminus(Terminus2):
    def _init_llm(self,**kwargs):
        return MemoryLLM(self.logs_dir.parent.name,kwargs['model_name'].split('/')[-1])
