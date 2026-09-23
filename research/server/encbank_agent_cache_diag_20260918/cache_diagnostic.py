"""Bounded cache correctness diagnosis on saved public interaction inputs; no scoring."""
from pathlib import Path
import gc, hashlib, json, os, platform, sys, traceback
H = Path(__file__).resolve().parent
SOURCE = H.parent / 'encbank_agent_ctx_r3_20260918'
sys.path.insert(0, str(SOURCE))
from common import load_model, tokenizer, MODELS, torch, dump

def reset(model):
    for obj in (model, model.model):
        if hasattr(obj, 'rope_deltas'): obj.rope_deltas = None

def compare(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    pa, pb = a.log_softmax(-1), b.log_softmax(-1)
    return dict(max_abs=float((a-b).abs().max()), rms=float((a-b).square().mean().sqrt()),
        top1_equal=bool(a.argmax() == b.argmax()), top1_ids=[int(a.argmax()), int(b.argmax())],
        top10_overlap=len(set(a.topk(10).indices.tolist()) & set(b.topk(10).indices.tolist()))/10,
        kl_a_b=float((pa.exp()*(pa-pb)).sum()), finite=bool(a.isfinite().all() and b.isfinite().all()))

try:
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    free, total = torch.cuda.mem_get_info(); assert free >= 70*2**30
    torch.cuda.set_per_process_memory_fraction(80*2**30/total)
    torch.set_num_threads(2); torch.manual_seed(20260917)
    tok=tokenizer(MODELS[1]); model=load_model(MODELS[1]); model.requires_grad_(False)
    assert all(p.device.type == 'cuda' for p in model.parameters())
    def enc(t): return tok.encode(t, add_special_tokens=False)
    def render(m,g=False): return tok.apply_chat_template(m,tokenize=False,add_generation_prompt=g,enable_thinking=False)
    def tensor(t): return torch.tensor([t],device='cuda',dtype=torch.long)
    box=SOURCE/'run_dense/mailbox'; requests=[json.loads(f.read_text()) for f in sorted(box.glob('*.request.json'))]
    rows=[]
    for task in dict.fromkeys(r['task_id'] for r in requests):
        matching=[f for f in sorted(box.glob('*.request.json')) if json.loads(f.read_text())['task_id']==task]
        if len(matching)<2: continue
        first, second=[json.loads(f.read_text()) for f in matching[:2]]
        response=json.loads(matching[0].with_name(matching[0].name.replace('.request.','.response.')).read_text())
        generated=response['generated_ids']; initial=render(first['messages']); withgen=render(first['messages'],True)
        assert withgen.startswith(initial); prefix=enc(withgen[len(initial):]); static=enc(initial)
        imend=tok.convert_tokens_to_ids('<|im_end|>'); closure=[] if generated[-1]==imend else [imend]
        ledger=static+prefix+generated+closure+enc('\n'+render([second['messages'][-1]]))+prefix
        reset(model)
        with torch.inference_mode():
            result=model(input_ids=tensor(static+prefix),use_cache=True,logits_to_keep=1)
            cache=result.past_key_values; del result
            for token in generated[:-1]:
                result=model(input_ids=tensor([token]),past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=result.past_key_values; del result
            old=len(static)+len(prefix)+len(generated)-1
            assert cache.get_seq_length()==old
            result=model(input_ids=tensor(ledger[old:]),attention_mask=torch.ones((1,len(ledger)),device='cuda',dtype=torch.long),past_key_values=cache,use_cache=True,logits_to_keep=1)
            inc=result.logits.clone(); del result,cache
            reset(model)
            result=model(input_ids=tensor(ledger),use_cache=True,logits_to_keep=1)
            fresh=result.logits.clone(); del result
            reset(model)
            nocache=model(input_ids=tensor(ledger),use_cache=False,logits_to_keep=1).logits
            row=dict(task_id=task,prompt_tokens=len(ledger),ledger_sha256=hashlib.sha256(json.dumps(ledger).encode()).hexdigest(),
                incremental_vs_fresh=compare(inc,fresh),incremental_vs_nocache=compare(inc,nocache),fresh_vs_nocache=compare(fresh,nocache),
                source_files={str(f.relative_to(SOURCE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [matching[0],matching[1],matching[0].with_name(matching[0].name.replace('.request.','.response.'))]})
            rows.append(row); dump(H/'diagnostic_progress.json',dict(rows=rows)); del inc,fresh,nocache
        gc.collect(); torch.cuda.empty_cache()
    dump(H/'diagnostic_result.json',dict(status='complete',job=os.environ['SLURM_JOB_ID'],host=platform.node(),gpu=torch.cuda.get_device_name(),
        rows=rows,quality_result=False,thresholds_changed=False,interpretation='Diagnostic comparison only; no benchmark answers regenerated or scored.'))
except BaseException:
    dump(H/'diagnostic_failure.json',dict(traceback=traceback.format_exc())); raise
