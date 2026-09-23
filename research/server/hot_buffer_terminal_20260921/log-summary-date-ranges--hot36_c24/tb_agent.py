"""Official Terminus2, with an isolated shared-filesystem model mailbox."""
import asyncio,json,os,time
from pathlib import Path
import harbor_unbounded
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM,LLMResponse,ContextLengthExceededError
from harbor.models.metric import UsageInfo
from common import ROOT,PLAN,save,sha
class AgentLLM(BaseLLM):
    def __init__(self,trial):
        self.task=os.environ['BAND_TASK'];self.arm=os.environ['BAND_ARM'];self.trial=trial;self.step=0;self.failed=False
        self.box=ROOT/'mailbox'/self.task;self.previous=[]
    def get_model_context_limit(self):return PLAN['context_tokens']
    def get_model_output_limit(self):return None
    async def call(self,prompt,**kwargs):
        assert not self.failed
        history=kwargs.get('message_history') or []
        messages=history+[dict(role='user',content=prompt)]
        if self.previous:assert history[:-1]==self.previous,'Conversation history changed'
        rid=f'{self.arm}_{self.step:05d}';req=self.box/(rid+'.request.json');resp=self.box/(rid+'.response.json')
        assert not req.exists(),'No duplicate model request'
        save(req,dict(request_id=rid,task=self.task,arm=self.arm,step=self.step,trial=self.trial,messages=messages,created_epoch=time.time()))
        start=time.monotonic()
        try:
            while not resp.exists():
                fail=ROOT/'pairs'/self.task/'worker_failure.json'
                if fail.exists():raise RuntimeError('Model worker failed: '+fail.read_text())
                if (ROOT/'pairs'/self.task/'parent_exit.json').exists():raise RuntimeError('Model worker exited before response')
                await asyncio.sleep(.05)
            result=json.loads(resp.read_text());assert result['request_sha256']==sha(req)
            if result['status']=='context_limit':raise ContextLengthExceededError(result.get('error','Physical context capacity'))
            if result['status']!='ok':raise RuntimeError(json.dumps(result))
        except BaseException:
            self.failed=True;save(self.box/(rid+'.cancel.json'),dict(at_epoch=time.time()));raise
        self.previous=messages;self.step+=1
        return LLMResponse(content=result['text'].strip(),model_name='Qwen3-8B/COMem-'+self.arm,
            usage=UsageInfo(prompt_tokens=result['prompt_tokens'],completion_tokens=result['generated_tokens'],cache_tokens=result.get('events',[{}])[0].get('hit_chunks',0)*512,cost_usd=0),
            prompt_token_ids=result['prompt_ids'],completion_token_ids=result['generated_ids'],
            extra={**{k:v for k,v in result.items() if k not in ['text','prompt_ids','generated_ids']},'transport_roundtrip_seconds':time.monotonic()-start})
class JointBandTerminus(Terminus2):
    def _init_llm(self,**kwargs):return AgentLLM(self.logs_dir.parent.name)
