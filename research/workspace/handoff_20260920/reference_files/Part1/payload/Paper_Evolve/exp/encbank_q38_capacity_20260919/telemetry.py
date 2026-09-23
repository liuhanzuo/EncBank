import ctypes,threading,time
class Nvml:
    class Memory(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    def __init__(self,uuid):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1');assert self.lib.nvmlInit_v2()==0
        self.handle=ctypes.c_void_p();assert self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(),ctypes.byref(self.handle))==0
        self.stop_event=threading.Event();self.samples=[]
    def sample(self):
        value=self.Memory();assert self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(value))==0
        self.samples.append(dict(at=time.perf_counter(),used_bytes=value.used,total_bytes=value.total))
    def loop(self):
        while not self.stop_event.wait(.1):self.sample()
    def start(self):self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def stop(self):self.stop_event.set();self.thread.join();self.sample();return max(x['used_bytes'] for x in self.samples)
