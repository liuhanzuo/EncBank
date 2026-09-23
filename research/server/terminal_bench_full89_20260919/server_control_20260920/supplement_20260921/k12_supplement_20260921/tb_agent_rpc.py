"""Official Terminus2 with isolated task histories and a shared model service."""
import asyncio, hashlib, json, time
from pathlib import Path
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM, LLMResponse, ContextLengthExceededError
from harbor.models.metric import UsageInfo
H=Path(__file__).resolve().parent
P=json.loads((H/'plan.json').read_text())
RPC=Path(P['rpc_root'])
from server_transport import save
class MailboxLLM(BaseLLM):
    def __init__(self,trial):
        self.trial=trial;self.task=trial.split('__')[0]
        self.task_id='tb_'+hashlib.sha256(trial.encode()).hexdigest()[:24]
        self.step=0;self.start=None;self.failed=False;self.last_messages=[]
        self.limit=None  # No cumulative task deadline under the newly authorized protocol.
    def get_model_context_limit(self):return P['context_tokens']
    def get_model_output_limit(self):return P['max_new_tokens']
    async def call(self,prompt,**kwargs):
        assert not self.failed,'No duplicate generation after failure'
        history=kwargs.get('message_history') or []
        messages=history+[dict(role='user',content=prompt)]
        if self.last_messages:assert history[:-1]==self.last_messages,'Prior task history changed'
        if self.start is None:self.start=time.monotonic()
        rid=f'{self.task_id}_{self.step:05d}';box=RPC/'comem';box.mkdir(parents=True,exist_ok=True)
        req=box/(rid+'.request.json');resp=box/(rid+'.response.json');err=box/(rid+'.error.json')
        assert not req.exists(),'Refuse duplicate request'
        began=time.monotonic()
        save(req,dict(request_id=rid,task_id=self.task_id,task=self.task,trial=self.trial,step=self.step,messages=messages,
            remaining_seconds=P['per_call_transport_watchdog_seconds'],client_created_epoch=time.time()))
        try:
            while not resp.exists():
                if err.exists():raise RuntimeError(json.loads(err.read_text())['error'])
                if time.monotonic()-began>P['per_call_transport_watchdog_seconds']+240:raise TimeoutError('Server controller response watchdog; no request replay')
                await asyncio.sleep(.1)
            r=json.loads(resp.read_text())
        except asyncio.CancelledError:
            self.failed=True;save(box/(rid+'.cancel.json'),dict(cancelled=True,at_epoch=time.time()));raise
        except BaseException:self.failed=True;raise
        if r['status']=='context_limit':raise ContextLengthExceededError('Full context exceeds registered native context; no crop')
        if r['status'] in ['deadline','cancelled']:
            self.failed=True
            raise RuntimeError('Per-call infrastructure watchdog or cancellation; incomplete task, no quality zero')
        if r['status']!='ok':self.failed=True;raise RuntimeError(json.dumps(r))
        text=r['text'];reasoning=None
        if '</think>' in text:reasoning,text=text.split('</think>',1)
        else:reasoning,text=text,''
        self.last_messages=messages;self.step+=1
        r['transport_roundtrip_seconds']=time.monotonic()-began
        return LLMResponse(content=text.strip(),reasoning_content=reasoning,model_name='Qwen3.8-27B/comem-native-j21',
            usage=UsageInfo(prompt_tokens=r['prompt_tokens'],completion_tokens=r['generated_tokens'],cache_tokens=r.get('cached_prefix_tokens',0),cost_usd=0),
            prompt_token_ids=r['prompt_ids'],completion_token_ids=r['generated_ids'],
            extra={k:v for k,v in r.items() if k not in ['text','prompt_ids','generated_ids']})
class ConcurrentCoMemTerminus(Terminus2):
    def _init_llm(self,**kwargs):return MailboxLLM(self.logs_dir.parent.name)
