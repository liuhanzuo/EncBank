"""Bounded generation diagnostics for the user-authorized model service."""
import concurrent.futures, datetime, json, time, urllib.request, urllib.error, re
from probe_judge_service import BASES, ROOT, conversation_key, NoRedirect, request

def stream(model,key):
    payload=dict(model=model,input=[{'role':'user','content':[{'type':'input_text','text':'Reply with exactly the word CORRECT.'}]}],
                 stream=True,store=False,reasoning={'effort':'low'})
    req=urllib.request.Request(BASES[0]+'/responses',data=json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'text/event-stream'})
    start=time.monotonic();out=dict(model=model,route='responses',stream=True,events=[],text='')
    try:
        with urllib.request.build_opener(NoRedirect).open(req,timeout=50) as resp:
            out['http_status']=resp.status
            for raw in resp:
                line=raw.decode('utf-8','replace').strip()
                if not line.startswith('data:'):continue
                data=line[5:].strip()
                if data=='[DONE]':break
                try:event=json.loads(data)
                except ValueError:continue
                kind=event.get('type')
                if kind and kind not in out['events']:out['events'].append(kind)
                if kind=='response.output_text.delta':out['text']+=event.get('delta','')
                if kind in ('response.completed','response.failed','response.incomplete'):
                    r=event.get('response',{})
                    out.update(response_status=r.get('status'),returned_model=r.get('model'),usage=r.get('usage'),error=r.get('error'))
                    break
                if kind=='error':out['error']=event;break
                if time.monotonic()-start>90:out['timeout']=True;break
    except urllib.error.HTTPError as e:
        raw=e.read(4000).decode('utf-8','replace').replace(key,'[REDACTED]')
        out.update(http_status=e.code,response_preview=re.sub(r'sk-[A-Za-z0-9_-]{24,}','[REDACTED]',raw)[:1500])
    except Exception as e:out['error_type']=type(e).__name__
    out['elapsed_s']=time.monotonic()-start
    return out

def main():
    key=conversation_key()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:
        sf=p.submit(stream,'gpt-5.6-sol',key)
        cf=p.submit(request,BASES[0],'/chat/completions',key,
            {'model':'gpt-5.5','messages':[{'role':'user','content':'Reply with exactly CORRECT.'}]},45)
        result={'streaming_sol':sf.result(),'minimal_chat_gpt55':cf.result()}
    result['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (ROOT/'judge_stream_probe.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':main()
