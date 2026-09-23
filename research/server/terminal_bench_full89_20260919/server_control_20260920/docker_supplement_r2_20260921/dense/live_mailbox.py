"""Run-scoped transport; model state stays on GPU, JSON is durably saved by the local owner.

The former per-reply BeeGFS create/rename path failed with errno121. This
transport uses the existing SSH control route plus an authenticated node-local
service. No generated reply is recomputed on a transport failure.
"""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from types import SimpleNamespace
import base64, fnmatch, hashlib, hmac, json, os, secrets, threading, time

def payload(value):return (json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode('utf8')

class MemoryPath:
    def __init__(self, box, name=''):self.box=box;self.name=name
    @property
    def stem(self):return Path(self.name).stem
    def __lt__(self,other):return self.name<other.name
    def __truediv__(self,name):return MemoryPath(self.box,name)
    def __str__(self):return str(self.box.root/self.name)
    def with_name(self,name):return MemoryPath(self.box,name)
    def exists(self):
        with self.box.lock:return self.name in self.box.files
    def read_bytes(self):
        with self.box.lock:return self.box.files[self.name][1]
    def read_text(self,*args,**kwargs):return self.read_bytes().decode('utf8')
    def stat(self):
        with self.box.lock:return SimpleNamespace(st_mtime_ns=self.box.files[self.name][0])
    def glob(self,pattern):
        with self.box.lock:return [MemoryPath(self.box,n) for n in self.box.files if fnmatch.fnmatchcase(n,pattern)]
    def write_immutable(self,value):
        data=payload(value)
        with self.box.lock:
            if self.name in self.box.files:
                assert self.box.files[self.name][1]==data,'Refuse a different completed payload'
            else:
                self.box.files[self.name]=(time.time_ns(),data)
                if self.name.endswith('.response.json'):
                    status=value['status'];self.box.response_counts[status]=self.box.response_counts.get(status,0)+1
                    self.box.generated_tokens+=value.get('generated_tokens',0)

class Mailbox:
    def __init__(self,root,status):
        self.root=Path(root);self.status=status;self.files={};self.lock=threading.RLock();self.events=[];self.event_disk_errors=[]
        self.response_counts={};self.generated_tokens=0
        self.token=secrets.token_hex(32);box=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                try:
                    if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+box.token):
                        self.send_error(403);return
                    n=int(self.headers['Content-Length']);assert 0<n<=32*2**20
                    q=json.loads(self.rfile.read(n));out=box.dispatch(q)
                    raw=json.dumps(out).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
                except Exception as exc:
                    raw=json.dumps(dict(error=type(exc).__name__,message=str(exc))).encode()
                    self.send_response(500);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        self.server=ThreadingHTTPServer(('0.0.0.0',0),Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
    def endpoint(self):return dict(host=os.environ.get('SLURMD_NODENAME','127.0.0.1'),port=self.server.server_port,token=self.token)
    def path(self):return MemoryPath(self)
    def event(self,kind,**kw):
        with self.lock:self.events.append(dict(event=kind,epoch=time.time(),**kw))
    def dispatch(self,q):
        action=q['action'];B=self.path()
        if action=='status':
            (B/'transport_probe.json').write_immutable(dict(connected=True))
            out=self.status()
            with self.lock:
                start=int(q.get('event_cursor',0));out['transport']=dict(version=1,event_cursor=len(self.events),events=self.events[start:],stored_files=len(self.files),response_counts=dict(self.response_counts),generated_tokens=self.generated_tokens,disk_event_errors=self.event_disk_errors)
            return out
        if action=='stop':
            B.__truediv__('stop.json').write_immutable(dict(reason='server Harbor parents closed'));return dict(stop_requested=True)
        rid=q['id'];assert rid.replace('_','').isalnum()
        if action=='publish':
            d=q['request'];assert d['request_id']==rid and not (B/(rid+'.request.json')).exists()
            status=self.status();assert 'worker_ready.json' in status and 'worker_failure.json' not in status
            remaining=d.pop('remaining_seconds');assert 0<=remaining<=12000
            d.update(published_epoch=time.time());d['deadline_epoch']=d['published_epoch']+remaining
            (B/(rid+'.request.json')).write_immutable(d);return dict(published=True)
        if action in ['release','cancel']:
            value=dict(task_id=rid,reason='owned Harbor task parent closed') if action=='release' else dict(cancelled=True)
            (B/(rid+'.'+action+'.json')).write_immutable(value);return {action+'_requested':True}
        if action=='wait':
            req=json.loads((B/(rid+'.request.json')).read_text());resp=B/(rid+'.response.json')
            while not resp.exists():
                assert 'worker_failure.json' not in self.status(),'Worker failed before response'
                if time.time()>req['deadline_epoch']+90:raise TimeoutError('Worker did not close expired generation')
                time.sleep(.1)
            raw=resp.read_bytes()
            return dict(path=str(resp),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),payload_base64=base64.b64encode(raw).decode(),transport='cluster-http-memory-v1')
        raise ValueError(action)

def client(endpoint,query,timeout=60):
    # Disable ambient HTTP proxies; the control request stays on the cluster.
    import urllib.request
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req=Request('http://'+endpoint['host']+':'+str(endpoint['port'])+'/',data=json.dumps(query).encode(),headers={'Authorization':'Bearer '+endpoint['token'],'Content-Type':'application/json'})
    with opener.open(req,timeout=timeout) as response:return json.loads(response.read())
