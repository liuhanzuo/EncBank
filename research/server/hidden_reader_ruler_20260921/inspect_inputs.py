import hashlib,json
from pathlib import Path
E=Path('/srv/encbank/qcomem_align_codex_20260911/repo/paper_autonomous_multifork_iteration/evidence')
paths=[
('single8k','honly_fp16_ruler8k_pair_preparation_20260912/single8k/inputs'),
('multikey8k','honly_ruler_multikey_preparation_20260912/inputs'),
('vt8k','honly_ruler_parallel_preparation_20260912/inputs'),
('single32k','honly_ruler_parallel_preparation_20260912/single32k/inputs'),
('multikey32k','honly_ruler_parallel_preparation_20260912/multikey32k/inputs'),
('vt32k','honly_ruler_parallel_preparation_20260912/vt32k/inputs'),
('single128k','honly_ruler_remaining64_128_preparation_20260913/single128k/inputs'),
('multikey128k','honly_ruler_remaining64_128_preparation_20260913/multikey128k/inputs'),
('vt128k','honly_ruler_remaining64_128_preparation_20260913/vt128k/inputs')]
records=[]
for name,p in paths:
    root=E/p;f=root/'inference_fixture.json'
    if not f.exists():print(name,'MISSING',str(f));continue
    obj=json.loads(f.read_text());items=obj if isinstance(obj,list) else obj.get('items',obj.get('documents',[]))
    print(name,'root_keys',list(obj)[:12] if isinstance(obj,dict) else 'list','n',len(items))
    first=items[0] if items else {}
    print(' item_summary', {k:(len(v) if isinstance(v,list) else v) for k,v in first.items() if k not in ['full_prompt','text']})
    labelpaths=list((root/'scoring_only').glob('*labels.json'))
    print(' labels', [str(p) for p in labelpaths])
    for labels in labelpaths:
        lab=json.loads(labels.read_text())
        print(' label_schema',str(lab)[:550])
    manifests=list(root.glob('*manifest.json'))+list(root.glob('generation_plan.json'))
    for mf in manifests:
        m=json.loads(mf.read_text());print(' manifest',mf.name,str(m)[:700])
    records.append(dict(cell=name,fixture=str(f),fixture_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
        labels=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in labelpaths],items=len(items)))
Path(__file__).with_name('input_inventory.json').write_text(json.dumps(records,indent=2)+'\n')
