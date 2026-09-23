"""Local read-only progress including actual process identity; no GPU allocations."""
import json,subprocess
from pathlib import Path
import config
def main():
    status=json.loads((config.ROOT/'status.json').read_text()) if (config.ROOT/'status.json').exists() else {}
    cmd=['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { $_.CommandLine -Match 'comem_bge_matched_20260918' } | Select-Object ProcessId,ParentProcessId,CommandLine,CreationDate | ConvertTo-Json -Compress"]
    result=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
    assert result.returncode==0,result.stderr
    processes=json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(processes,dict):processes=[processes]
    live=any(p['ProcessId']==status.get('pid') and 'controller.py' in p['CommandLine'] for p in processes)
    value=dict(status=status,controller_alive=live,processes=processes)
    for phase in ['calibration','extended','quality']:
        value[phase]=[]
        for p in sorted((config.ROOT/phase).glob('process_*')):
            row=dict(path=str(p))
            for name in ['progress','complete','parent_exit']:
                file=p/(name+'.json');row[name]=json.loads(file.read_text()) if file.exists() else None
            if row['parent_exit'] and row['parent_exit']['returncode']!=0:
                err=p/'stderr.log';row['error_tail']=err.read_text(encoding='utf-8',errors='replace')[-5000:] if err.exists() else None
            value[phase].append(row)
    for name in ['budget_decision','summary','reservation_release']:
        p=config.ROOT/(name+'.json');value[name]=json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    print(json.dumps(value,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
