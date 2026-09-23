"""Freeze an independent Hot24 PEFT-merged real Terminal-Bench arm."""
import ast,hashlib,json,shutil,time
from pathlib import Path
B=Path('/srv/encbank/qcomem_align_codex_20260911')
SRC=B/'tf27b_hot_live_20260921/hot24'
R=B/'tf27b_hot_live_20260921/hot24_peft_merged'
PILOT=B/'terminal_bench_full89_20260919/server_control_20260920/comem_vllm_pilot_20260922/attempts/peft_r1'
assert not R.exists(),'Never overwrite or duplicate an existing experiment'
manifest=json.loads((SRC/'source_manifest.json').read_text())
for name,d in manifest.items():assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==d,name
R.mkdir()
for name in manifest:
    if name in ['plan.json','hot24_preflight.json']:continue
    shutil.copyfile(SRC/name,R/name)
shutil.copyfile(PILOT/'comem_peft.py',R/'comem_peft.py')
p=json.loads((SRC/'plan.json').read_text())
plan=dict(p,experiment='Qwen3.8-27B Transformers Hot24 PEFT-merged independent Terminal-Bench arm',
    variant='hot24_peft_merged',lora_execution='peft_merged_bf16',
    lora='suffix j21 rank32 alpha32 final4000, PEFT safe_merge into BF16 backbone',
    peft_adapter=str(PILOT/'peft_adapter'),
    task_concurrency=24,decode_batch_size=24,hot_chunks=24,
    authorization='User 2026-09-22: use the 24-way Hot Buffer version with PEFT-merged LoRA as an additional real Terminal-Bench experiment; confirm actual start.',
    benchmark_accounting='Independent new 32-task Hot24-merged series; never pooled with old Hot24/Hot32 or full89 results',
    qualification_boundary='Original frozen core qualification retained; loader/weights explicitly changed by user. New merged startup checks required; no claim of unchanged legacy logits.')
assert len(plan['tasks'])==32 and plan['tasks']==p['tasks']
(R/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')

path=R/'model_setup.py';source=path.read_text()
old="resolve_configs(plan,identity,saved);reader.attach();load_state(reader,saved,identity);del saved"
assert source.count(old)==1
new="""resolve_configs(plan,identity,saved);del saved
        from comem_peft import load_comem_peft, file_sha
        assert plan['lora_execution']=='peft_merged_bf16'
        merge_start=time.perf_counter()
        model,peft_owner,targets=load_comem_peft(model,plan['peft_adapter'],mode='merged_bf16')
        target_names=sorted(targets)
        assert len(target_names)==333 and not any('lora_' in n for n,_ in model.named_parameters())
        assert not any(type(m).__name__ in ['LoRALinear','LegacyArithmetic'] for m in model.modules())
        del peft_owner,targets
        reader=HybridReader(model,21)
        torch.cuda.synchronize()
        save(ROOT/'peft_merge_receipt.json',dict(epoch=time.time(),execution='peft_merged_bf16',
            method='PEFT merge_and_unload(safe_merge=True)',modules=len(target_names),
            targets_sha256=hashlib.sha256(json.dumps(target_names,separators=(',',':')).encode()).hexdigest(),
            seconds=time.perf_counter()-merge_start,legacy_bitwise_equivalence_claimed=False,
            converted_adapter_files=json.loads((Path(plan['peft_adapter'])/'conversion_receipt.json').read_text())['files'],
            merged_checkpoint_saved=False))"""
path.write_text(source.replace(old,new))
path=R/'worker.py';source=path.read_text()
old="for n in ['hybrid_hot.py','hybrid_reader.py','batch_cache.py','model_setup.py']:"
assert source.count(old)==1
source=source.replace(old,"for n in ['hybrid_hot.py','hybrid_reader.py','batch_cache.py']:")
old="model,r,tok,stop=load(P,adapter=P['arm']!='dense')"
assert source.count(old)==1
new="""assert P['lora_execution']=='peft_merged_bf16'
    assert json.loads((ROOT/'peft_preflight.json').read_text())['passed']
    model,r,tok,stop=load(P,adapter=P['arm']!='dense')
    from merged_qualification import qualify
    qualify(r,tok,P)"""
source=source.replace(old,new)
source=source.replace("arm=P['arm'],max_decode_batch=", "arm=P['arm'],variant=P['variant'],lora_execution=P['lora_execution'],max_decode_batch=")
source=source.replace("engine='transformers',arm=P['arm'],cached_prefix_tokens=", "engine='transformers',arm=P['arm'],variant=P['variant'],lora_execution=P['lora_execution'],cached_prefix_tokens=")
path.write_text(source)
for name in ['jobs','results','qualification','mailbox','logs','tmp','cache','pairs','configs','worker']:(R/name).mkdir()
for task in plan['tasks']:
    (R/'pairs'/task).mkdir();(R/'mailbox'/task).mkdir()
for f in R.glob('*.py'):ast.parse(f.read_text())
for path,expected in json.loads((R/'task_manifest.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
proof=dict(epoch=time.time(),source=str(SRC),root=str(R),tasks_unchanged=True,resources_unchanged=True,
    predecessor_source_manifest_sha256=hashlib.sha256((SRC/'source_manifest.json').read_bytes()).hexdigest(),
    unchanged_sources=[n for n in manifest if (R/n).exists() and hashlib.sha256((R/n).read_bytes()).hexdigest()==manifest[n]],
    modified_sources=['model_setup.py','worker.py','plan.json'],new_sources=['comem_peft.py','merged_qualification.py'],
    reason='Explicit user-requested merged-LoRA experiment; startup qualification revised to distinguish identical core from new weight execution.')
(R/'variant_diff.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
