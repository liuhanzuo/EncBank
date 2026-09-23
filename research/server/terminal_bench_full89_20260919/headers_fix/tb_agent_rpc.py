"""Official Terminus2 with isolated task histories and a shared model service."""
import asyncio, hashlib, json, time
from pathlib import Path
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM, LLMResponse, ContextLengthExceededError
from harbor.models.metric import UsageInfo
H=Path(__file__).resolve().parent
P=json.loads((H/'plan.json').read_text())
RPC=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_full89_20260919/local_rpc_headers_fix')
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=True)+'\n');t.replace(p)
class MailboxLLM(BaseLLM):
    def __init__(self,trial):
        self.trial=trial;self.task=trial.split('__')[0]
        self.task_id='tb_'+hashlib.sha256(trial.encode()).hexdigest()[:24]
        self.step=0;self.start=None;self.failed=False;self.last_messages=[]
        self.limit=next(x['agent_seconds'] for x in P['timeout_inventory'] if x['task']==self.task)
    def get_model_context_limit(self):return P['context_tokens']
    def get_model_output_limit(self):return P['max_new_tokens']
    async def call(self,prompt,**kwargs):
        assert not self.failed,'No duplicate generation after failure'
        history=kwargs.get('message_history') or []
        messages=history+[dict(role='user',content=prompt)]
        if self.last_messages:assert history[:-1]==self.last_messages,'Prior task history changed'
        if self.start is None:self.start=time.monotonic()
        rid=f'{self.task_id}_{self.step:05d}';box=RPC/'dense';box.mkdir(parents=True,exist_ok=True)
        req=box/(rid+'.request.json');resp=box/(rid+'.response.json');err=box/(rid+'.error.json')
        assert not req.exists(),'Refuse duplicate request'
        began=time.monotonic()
        save(req,dict(request_id=rid,task_id=self.task_id,task=self.task,trial=self.trial,step=self.step,messages=messages,
            remaining_seconds=max(0,self.limit-(began-self.start)),client_created_epoch=time.time()))
        try:
            while not resp.exists():
                if err.exists():raise RuntimeError(json.loads(err.read_text())['error'])
                await asyncio.sleep(.1)
            r=json.loads(resp.read_text())
        except asyncio.CancelledError:
            self.failed=True;save(box/(rid+'.cancel.json'),dict(cancelled=True,at_epoch=time.time()));raise
        except BaseException:self.failed=True;raise
        if r['status']=='context_limit':raise ContextLengthExceededError('Full context exceeds registered native context; no crop')
        if r['status']!='ok':self.failed=True;raise RuntimeError(json.dumps(r))
        text=r['text'];reasoning=None
        if '</think>' in text:reasoning,text=text.split('</think>',1)
        else:reasoning,text=text,''
        self.last_messages=messages;self.step+=1
        r['transport_roundtrip_seconds']=time.monotonic()-began
        return LLMResponse(content=text.strip(),reasoning_content=reasoning,model_name='Qwen3.8-27B/dense-vllm',
            usage=UsageInfo(prompt_tokens=r['prompt_tokens'],completion_tokens=r['generated_tokens'],cache_tokens=r.get('cached_prefix_tokens',0),cost_usd=0),
            prompt_token_ids=r['prompt_ids'],completion_token_ids=r['generated_ids'],
            extra={k:v for k,v in r.items() if k not in ['text','prompt_ids','generated_ids']})
class ConcurrentDenseTerminus(Terminus2):
    def _init_llm(self,**kwargs):return MailboxLLM(self.logs_dir.parent.name)
