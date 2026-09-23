"""Attach handles to surviving owned task launchers; never restart their worlds."""
from pathlib import Path
import ctypes,json,os
from ctypes import wintypes
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
class Adopted:
    def __init__(self,pid):
        self.pid=pid;self.handle=k.OpenProcess(0x100000|0x1000,False,pid)
        self.original_parent=False
    def poll(self):
        if not self.handle:return 'unavailable_after_original_parent_exit'
        code=wintypes.DWORD();assert k.GetExitCodeProcess(self.handle,ctypes.byref(code))
        return None if code.value==259 else int(code.value)
    def wait(self):
        if self.handle:k.WaitForSingleObject(self.handle,0xffffffff)
        return self.poll()
def recover(H,O,BOX,RESULTS,P,host_admission,save):
    previous=(O/'resume_registration.json') if (O/'resume_registration.json').exists() else (O/'owner_registration.json')
    old=json.loads(previous.read_text());oldproc=Adopted(old['pid'])
    assert oldproc.poll() is not None,'Original owner still active'
    assert (O/'controller_failure.json').exists()
    save(O/'resume_registration.json',dict(pid=os.getpid(),old_pid=old['pid'],mode='adopt existing task processes, no duplicate launch'))
    completed=[p.stem for p in (O/'receipts').glob('*.json')];active={}
    launched=set()
    for p in (O/'launches').glob('*.json'):
        q=json.loads(p.read_text());name=q['task'];launched.add(name)
        if name in completed:continue
        active[name]=dict(task=name,memory_mb=q['memory_mb'],cpus=q['cpus'],child=Adopted(q['pid']),
            out=(O/(name+'.stdout.log')).open('ab'),err=(O/(name+'.stderr.log')).open('ab'),argv=q['argv'])
    with host_admission.locked():
        d=host_admission.read()
        for name in active:
            key=H.name+':'+name;assert d[key]['owner_pid']==old['pid'];d[key]['owner_pid']=os.getpid();d[key]['adopted_from_pid']=old['pid']
        host_admission.save(d)
    seen={p.name.replace('.response.','.request.') for p in BOX.glob('*.response.json')}
    seen|={p.name.replace('.error.','.request.') for p in BOX.glob('*.error.json')}
    pending=[r for r in P['resource_inventory'] if r['task'] not in launched]
    return active,completed,pending,seen
