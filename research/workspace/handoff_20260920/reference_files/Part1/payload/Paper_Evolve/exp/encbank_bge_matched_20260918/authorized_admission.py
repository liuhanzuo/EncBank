"""Campaign-only desktop allowance explicitly authorized by the user on 2026-09-19.

Keep the existing cross-campaign lock, Python exclusion, and allocator cap.
The shared gpu_gate.py on disk and every other process retain their default rules.
"""
import json,os,subprocess,sys,time
import config
sys.path.insert(0,'F:/Paper_Evolve/exp')
import gpu_gate

MAX_DESKTOP_GIB=7
MIN_FREE_GIB=24

def local_ready():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    used,free=map(int,r.stdout.strip().splitlines()[0].split(','))
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
    others=[x for x in r.stdout.splitlines() if 'python' in x.lower() and x.split(',')[0].strip()!=str(os.getpid())]
    return used<MAX_DESKTOP_GIB*1024 and free>=MIN_FREE_GIB*1024 and not others,dict(used_mib=used,free_mib=free,python_compute=others)

def acquire_gpu(**kwargs):
    auth=json.loads((config.ROOT/'desktop_admission_authorization.json').read_text(encoding='utf-8-sig'))
    assert auth['authorized'] and auth['campaign']==str(config.ROOT)
    previous=gpu_gate.HARD_IDLE_CEILING_GIB
    # This in-process override applies only to this explicitly authorized campaign.
    gpu_gate.HARD_IDLE_CEILING_GIB=MAX_DESKTOP_GIB
    try:
        result=gpu_gate.acquire_gpu(need_gb=MIN_FREE_GIB,idle_slack_gb=MAX_DESKTOP_GIB,**kwargs)
    finally:
        gpu_gate.HARD_IDLE_CEILING_GIB=previous
    stable=None;deadline=time.monotonic()+kwargs.get('max_wait',21600)
    while time.monotonic()<deadline:
        ready,observation=local_ready()
        stable=(stable or time.monotonic()) if ready else None
        if stable is not None and time.monotonic()-stable>=45:
            result.update(stable_seconds=time.monotonic()-stable,final_observation=observation,
                min_free_gib=MIN_FREE_GIB,authorization=str(config.ROOT/'desktop_admission_authorization.json'))
            return result
        time.sleep(5)
    raise TimeoutError('Authorized desktop admission did not stay free of other Python GPU jobs')
