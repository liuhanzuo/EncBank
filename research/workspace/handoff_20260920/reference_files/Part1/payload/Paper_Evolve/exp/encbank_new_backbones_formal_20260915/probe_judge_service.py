"""Query the two user-authorized endpoints without exposing the conversation key."""
import argparse, concurrent.futures, datetime, functools, hashlib, json, re, urllib.error, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
THREAD='01a094b1-4989-77d1-ba55-0df813ab68b3'
BASES=['https://sbtunnel.xiaoaojianghu.fun/v1', 'https://maas-openapi.wanjiedata.com/api/v1']

@functools.lru_cache(maxsize=1)
def conversation_key():
    # Only this conversation's user messages; never enumerate unrelated credentials.
    paths=list(Path('/srv/encbank/client/.codex/sessions').rglob('*'+THREAD+'*.jsonl'))
    found=[]
    for path in paths:
        with path.open(encoding='utf-8') as f:
            for line in f:
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                p=r.get('payload',{})
                if not isinstance(p,dict):continue
                texts=[]
                if r.get('type')=='response_item' and p.get('role')=='user':
                    texts=[c.get('text','') for c in p.get('content',[]) if isinstance(c,dict)]
                elif r.get('type')=='event_msg' and p.get('type')=='user_message':
                    texts=[p.get('message','')]
                for text in texts:
                    if not isinstance(text,str):continue
                    for key in re.findall(r'(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{24,}',text):
                        found.append((r.get('timestamp',''),key))
    assert found,'The provided credential is not yet available in this conversation log'
    ref=ROOT/'judge_credential_reference.json'
    if ref.exists():
        expected=json.loads(ref.read_text())
        matched=[k for ts,k in found if hashlib.sha256(k.encode()).hexdigest()==expected['sha256']]
        assert matched, 'The specifically authorized judge key is no longer available in this conversation'
        return matched[-1]
    timestamp,key=sorted(found,key=lambda x:x[0])[-1]
    # A non-secret identifier prevents accidentally using a later unrelated key.
    ref.write_text(json.dumps(dict(thread=THREAD,message_timestamp=timestamp,
        sha256=hashlib.sha256(key.encode()).hexdigest(),credential_value_stored=False),indent=2)+'\n')
    return key

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def request(base,path,key,payload=None,timeout=25):
    assert base in BASES, 'Only explicitly authorized providers may receive the key'
    data=json.dumps(payload).encode('utf-8') if payload is not None else None
    headers={'Accept':'application/json','User-Agent':'MidCache-Judge-Probe/1.0'}
    if key:headers['Authorization']='Bearer '+key
    if data is not None:headers['Content-Type']='application/json'
    req=urllib.request.Request(base+path,data=data,headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect).open(req,timeout=timeout) as resp:
            status=resp.status;response_headers=dict(resp.headers);raw=resp.read(2_000_000).decode('utf-8','replace')
    except urllib.error.HTTPError as e:
        status=e.code;response_headers=dict(e.headers);raw=e.read(20_000).decode('utf-8','replace')
    except Exception as e:
        return dict(ok=False,error_type=type(e).__name__)
    if key:raw=raw.replace(key,'[REDACTED]')
    raw=re.sub(r'sk-[A-Za-z0-9_-]{24,}','[REDACTED]',raw)
    try:body=json.loads(raw)
    except ValueError:return dict(ok=False,status=status,response_type='non-JSON',response_preview=raw[:1200],
        response_headers={k:v for k,v in response_headers.items() if k.lower() in ('server','content-type','via','x-cache','cf-ray')})
    return dict(ok=200<=status<300,status=status,body=body)

def main():
    key=conversation_key()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        values=list(pool.map(lambda b:(b,request(b,'/models',key)),BASES))
    report=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),credential_printed=False,endpoints={})
    for base,res in values:
        body=res.pop('body',{})
        if res['ok']:
            models=body.get('data',body.get('models',[])) if isinstance(body,dict) else []
            res['models']=[m.get('id',m.get('name')) if isinstance(m,dict) else m for m in models]
            res['models']=[m for m in res['models'] if isinstance(m,str)]
            res['model_count']=len(res['models'])
        else:
            err=body.get('error',{}) if isinstance(body,dict) else {}
            if isinstance(err,dict):res['provider_error']={k:str(err[k])[:400] for k in ('type','code','message') if k in err}
            elif isinstance(err,str):res['provider_error']=err[:400]
        report['endpoints'][base]=res
    (ROOT/'judge_service_models.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))

if __name__=='__main__':main()
