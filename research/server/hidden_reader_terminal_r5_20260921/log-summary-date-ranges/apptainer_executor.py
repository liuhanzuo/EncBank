"""Persistent command parent inside one task container (standard-library only)."""
import json, os, signal, socketserver, subprocess


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            request=json.loads(self.rfile.readline())
            env=dict(os.environ)
            env.update(request.get('env') or {})
            process=subprocess.Popen(['bash','-c',request['command']],cwd=request.get('cwd') or os.getcwd(),
                env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                text=True,start_new_session=True)
            try:
                stdout,stderr=process.communicate(timeout=request.get('timeout'))
                code=process.returncode
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                stdout,stderr=process.communicate()
                code=124
            result=dict(stdout=stdout,stderr=stderr,return_code=code)
        except Exception as error:
            result={'error':repr(error)}
        self.wfile.write((json.dumps(result)+'\n').encode())


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads=True


os.umask(0o077)
with Server('/staging/executor.sock',Handler) as server:
    server.serve_forever()
