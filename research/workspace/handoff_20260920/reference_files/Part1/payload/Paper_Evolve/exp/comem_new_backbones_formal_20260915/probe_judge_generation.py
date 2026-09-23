"""Test real generation on two listed judge candidates before selecting either."""
import concurrent.futures, datetime, json, time
from probe_judge_service import ROOT, BASES, conversation_key, request

def extract(body,route):
    if route=='chat/completions':
        return (body.get('choices') or [{}])[0].get('message',{}).get('content') or ''
    return ''.join(c.get('text','') for item in body.get('output',[]) if item.get('type')=='message'
                   for c in item.get('content',[]) if c.get('type')=='output_text')

def attempt(model,key):
    results=[]
    for route in ('chat/completions','responses'):
        if route=='chat/completions':
            payload=dict(model=model,messages=[dict(role='user',content='Reply with exactly the word CORRECT.')],
                         max_completion_tokens=128,reasoning_effort='none')
        else:
            payload=dict(model=model,input='Reply with exactly the word CORRECT.',max_output_tokens=128,
                         reasoning={'effort':'none'},store=False)
        start=time.monotonic();res=request(BASES[0],'/'+route,key,payload,timeout=45)
        body=res.pop('body',{})
        result=dict(model=model,route=route,elapsed_s=time.monotonic()-start,**res)
        if res['ok']:
            result.update(returned_model=body.get('model'),text=extract(body,route),usage=body.get('usage'),
                          response_status=body.get('status'),finish_reason=(body.get('choices') or [{}])[0].get('finish_reason'))
        else:result['error']=body.get('error',{}) if isinstance(body,dict) else {}
        results.append(result)
        if result.get('text','').strip()=='CORRECT':break
    return results

def main():
    key=conversation_key()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:
        runs=list(p.map(lambda m:attempt(m,key),('gpt-5.4-mini','gpt-5.5')))
    report=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_url=BASES[0],attempts=sum(runs,[]))
    (ROOT/'judge_generation_probe.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))

if __name__=='__main__':main()
