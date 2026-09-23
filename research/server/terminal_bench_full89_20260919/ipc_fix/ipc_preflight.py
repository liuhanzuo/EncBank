"""CPU-only exact-length IPC communication check; no torch/vLLM model imports."""
from pathlib import Path
import hashlib,json,os,socket,tempfile,time,uuid
import zmq
H=Path(__file__).resolve().parent
root=Path('/srv/encbank/qencbank_runtime_20260911/t89i3').resolve()
root.relative_to(Path('/srv/encbank').resolve());assert root.is_dir()
os.environ['TMPDIR']=os.environ['VLLM_RPC_BASE_PATH']=str(root);tempfile.tempdir=None
assert tempfile.gettempdir()==str(root)
src=Path('/srv/encbank/venvs/rllm/lib/python3.12/site-packages/vllm/utils/network_utils.py')
text=src.read_text();assert 'base_rpc_path = envs.VLLM_RPC_BASE_PATH' in text and 'return f"ipc://{base_rpc_path}/{uuid4()}"' in text
path=root/str(uuid.uuid4());assert len(os.fsencode(path))<=107
context=zmq.Context();a=context.socket(zmq.PAIR);b=context.socket(zmq.PAIR)
for s in [a,b]:s.setsockopt(zmq.RCVTIMEO,5000);s.setsockopt(zmq.SNDTIMEO,5000);s.setsockopt(zmq.LINGER,0)
try:
 a.bind('ipc://'+str(path));b.connect('ipc://'+str(path));a.send(b'dense-ipc-check');assert b.recv()==b'dense-ipc-check'
 b.send(b'ack');assert a.recv()==b'ack'
finally:a.close();b.close();context.term();path.unlink(missing_ok=True)
with tempfile.TemporaryDirectory(prefix='pymp-',dir=root) as d:
 unix=Path(d)/('listener-'+uuid.uuid4().hex[:8]);assert len(os.fsencode(unix))<=107
 server=socket.socket(socket.AF_UNIX);client=socket.socket(socket.AF_UNIX)
 try:
  server.settimeout(5);client.settimeout(5);server.bind(str(unix));server.listen(1);client.connect(str(unix));peer,_=server.accept()
  with peer:client.sendall(b'check');assert peer.recv(5)==b'check'
 finally:client.close();server.close()
out=dict(status='PASS',epoch=time.time(),gpu_model_calls=0,zmq_version=zmq.__version__,rpc_root=str(root),socket_path_bytes=len(os.fsencode(path)),max_socket_path_bytes=107,bidirectional_zmq=True,python_nested_unix_socket=True,vllm_network_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),actual_parent_wait_required=True)
(H/'ipc_preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
