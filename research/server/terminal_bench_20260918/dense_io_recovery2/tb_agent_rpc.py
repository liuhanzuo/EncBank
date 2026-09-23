"""Official Terminus2 native model backend; filesystem RPC to the Windows owner."""
import asyncio,json,time,uuid
from pathlib import Path
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM,LLMResponse,ContextLengthExceededError
from harbor.models.metric import UsageInfo
LOCAL=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_20260918/rpc_transport_metrics');LOCAL.mkdir(exist_ok=True)
RPC=Path('/srv/encbank/legacy_workspace/.runtime/terminal_bench_20260918/local_rpc_dense_io_recovery2')
def save(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=True)+'\n');t.replace(p)
async def control(arm,action,rid):
 assert action=='cancel';save(RPC/arm/(rid+'.cancel.json'),dict(cancelled=True));return dict(cancel_requested=True)
async def exchange(arm,d):
 rid=d['request_id'];box=RPC/arm;box.mkdir(parents=True,exist_ok=True);req=box/(rid+'.request.json');resp=box/(rid+'.response.json');err=box/(rid+'.error.json')
 assert not req.exists(),'Never repeat an already staged generation'
 save(req,d)
 while not resp.exists():
  if err.exists():raise RuntimeError(json.loads(err.read_text())['error'])
  await asyncio.sleep(.1)
 return json.loads(resp.read_text())
class MailboxLLM(BaseLLM):
 def __init__(self,arm):self.arm=arm;self.task_id='tb_'+uuid.uuid4().hex;self.step=0;self.start=None;self.last_messages=[];self.failed=False
 def get_model_context_limit(self):return 36864
 def get_model_output_limit(self):return 4096
 async def call(self,prompt,**kwargs):
  assert not self.failed,'Transport failure is terminal for this task; no duplicate generation'
  history=kwargs.get('message_history') or [];messages=history+[dict(role='user',content=prompt)]
  assert len(history)==2*self.step
  if self.last_messages:assert history[:-1]==self.last_messages,'Prior chat changed'
  if self.start is None:self.start=time.monotonic()
  rid=f'{self.task_id}_{self.step:05d}';began=time.monotonic()
  request=dict(action='call',request_id=rid,task_id=self.task_id,step=self.step,messages=messages,remaining_seconds=max(0,900-(began-self.start)))
  try:r=await exchange(self.arm,request)
  except asyncio.CancelledError:
   self.failed=True;await asyncio.shield(control(self.arm,'cancel',rid));raise
  except BaseException:self.failed=True;raise
  r['transport_roundtrip_seconds']=time.monotonic()-began
  (LOCAL/(rid+'.transport.json')).write_text(json.dumps(dict(seconds=r['transport_roundtrip_seconds'],arm=self.arm,task_id=self.task_id,step=self.step))+'\n')
  if r['status']=='context_limit':raise ContextLengthExceededError('Registered native context limit')
  if r['status']=='deadline':raise TimeoutError('Registered per-task agent deadline')
  text=r['text'];reasoning=None
  if '</think>' in text:reasoning,text=text.split('</think>',1)
  else:reasoning,text=text,''
  self.last_messages=messages;self.step+=1
  return LLMResponse(content=text.strip(),reasoning_content=reasoning,model_name='Qwen3.8-27B/'+self.arm,usage=UsageInfo(prompt_tokens=r['prompt_tokens'],completion_tokens=r['generated_tokens'],cache_tokens=min(r['prompt_tokens'],r.get('cached_prefix_tokens',0)),cost_usd=0),completion_token_ids=r['generated_ids'],extra={k:v for k,v in r.items() if k not in ['text','generated_ids']})
class QCoMemTerminus(Terminus2):
 def _init_llm(self,**kwargs):
  arm=kwargs['model_name'].split('/')[-1];assert arm in ['dense','raw_shared','comem'];return MailboxLLM(arm)
