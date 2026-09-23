"""Run correctness and training smoke in fresh processes; never report as science."""
import json,subprocess,sys
import config
from pathlib import Path

def main():
    out=config.ROOT/'smoke';out.mkdir(exist_ok=True)
    commands=[['test_cpu.py'],['train.py','--j','6','--seed','42','--smoke'],
        ['train.py','--j','24','--seed','42','--smoke'],['serving.py','--smoke']]
    records=[]
    for c in commands:
        cmd=[sys.executable,'-B','-u',str(config.ROOT/c[0]),*c[1:]]
        code=subprocess.Popen(cmd).wait();records.append(dict(command=cmd,actual_wait=True,returncode=code))
        (out/'processes.json').write_text(json.dumps(records,indent=2))
        assert code==0,cmd
    import torch
    a=torch.load(config.ROOT/'smoke_training/j6_s42/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(config.ROOT/'smoke_training/j24_s42/last.pt',map_location='cpu',weights_only=False)
    assert set(a['named'])==set(b['named'])
    assert all(a['named'][k].shape==b['named'][k].shape for k in a['named'])
    assert a['metadata']['trainable_parameters']==b['metadata']['trainable_parameters']
    assert a['metadata']['initial_adapter_sha256']==b['metadata']['initial_adapter_sha256']
    assert a['token_cursor']==b['token_cursor'] and a['step']==b['step']==2
    # Different loss/update is expected; equal tensors after training is NOT a requirement.
    (out/'complete.json').write_text(json.dumps(dict(complete=True,actual_wait=True,
        trainable_parameters=a['metadata']['trainable_parameters'],matched_layers_and_shapes=True,
        matched_training_token_cursor=True,formal_result=False)))

if __name__=='__main__':main()
