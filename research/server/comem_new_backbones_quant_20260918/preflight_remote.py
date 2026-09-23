"""Verify existing completed inputs once, without GPU or changing old artifacts."""
import json, os, time
from pathlib import Path
import torch
from common import ROOT, MODELS, TRAIN_DATA, dump
from binding import OLD, REMOTE, sha256, old_predictions, verify_code

def main():
    assert ROOT.resolve()==REMOTE and str(ROOT.resolve()).startswith('/srv/encbank/')
    verify_code()
    assert json.loads((ROOT/'cpu_checks.json').read_text())['passed']
    if (ROOT/'verified_inputs.json').exists():print('Existing verified inputs retained.');return
    evidence=dict(at=time.time(),models={},training_data_sha256=sha256(TRAIN_DATA))
    for cfg in MODELS:
        print('Verifying '+cfg['name'],flush=True)
        folder=Path(cfg['path']);assert str(folder.resolve()).startswith('/srv/encbank/')
        download=json.loads((folder/'download_complete.json').read_text())
        assert download['revision']==cfg['revision']
        model_files={};weight_files={}
        for p in sorted(folder.iterdir()):
            if not p.is_file() or p.suffix not in ('.json','.jinja','.txt','.safetensors'):continue
            checksum=sha256(p)
            if p.suffix=='.safetensors':
                meta=folder/'.cache/huggingface/download'/(p.name+'.metadata')
                assert meta.exists(),p
                revision,etag,*_=meta.read_text().splitlines()
                assert revision==cfg['revision'],(p,revision)
                assert len(etag)==64 and checksum==etag,(p,'weight digest differs from downloaded object')
                st=p.stat();weight_files[p.name]=dict(sha256=checksum,bytes=st.st_size,mtime_ns=st.st_mtime_ns,revision=revision)
                print('Verified weight '+p.name,flush=True)
            else:model_files[p.name]=checksum
        assert weight_files
        training=OLD/'training'/cfg['name']
        complete=json.loads((training/'complete.json').read_text())
        assert complete['complete'] and complete['steps']==4000 and complete['model']==cfg
        adapter=torch.load(training/'adapter-final.pt',map_location='cpu',weights_only=False)
        assert adapter['model']==cfg and adapter['j']==cfg['j'] and adapter['step']==4000
        assert adapter['rank']==adapter['alpha']==32 and adapter['modules']
        del adapter
        training_files={n:sha256(training/n) for n in ('adapter-final.pt','complete.json','protocol.json','pg19_tokens.u32','tokens.json')}
        prep=OLD/'samples'/cfg['name'];spec=json.loads((prep/'complete.json').read_text())
        assert spec['model']==cfg and spec['samples']==7236
        assert spec['totals']==dict(ruler=1500,longeval=500,longbench=1150,babilong=2100,locomo=1986)
        samples={p.name:sha256(p) for p in sorted(prep.glob('shard*.jsonl.gz'))};assert len(samples)==4
        samples['complete.json']=sha256(prep/'complete.json')
        pred=old_predictions(cfg);assert all(r['status']=='ok' for r in pred.values())
        outputs={}
        for base in ('results','results_locomo'):
            for p in sorted((OLD/base/cfg['name']).glob('shard*/predictions.jsonl')):
                outputs[str(p.relative_to(OLD))]=sha256(p)
        evidence['models'][cfg['name']]=dict(model=cfg,model_files=model_files,weight_files=weight_files,
             training_files=training_files,samples=samples,old_prediction_files=outputs,
             old_H16_records=len(pred),original_recipe=json.loads((training/'protocol.json').read_text()))
        dump(ROOT/'preflight_progress.json',dict(phase='verified_model',model=cfg['name'],at=time.time()))
    dump(ROOT/'verified_inputs.json',evidence)
    print('Preflight complete; original H16 generation only reusable after new wrapper parity passes.',flush=True)

if __name__=='__main__':main()
