"""Read-only, bounded diagnostics for three prompt/teacher-forced routing cases.

The original reader methods perform every computation up to _select. A private
exception stops execution after the original selection; no deep read/decode is
needed to determine a route. Instrumentation records tensors and operation
metadata without replacing arithmetic. This is not a performance benchmark.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'encbank_v2_benchmarks_20260908'))
sys.path.insert(0, str(HERE.parents[1] / 'Encbank'))

CASES = [
 ('B', 'qasper-train:1909.01013:03ce42ff53aa3f1775bc57e50012f6eb1998c480'),
 ('B', 'qasper-train:1909.06937:b6858c505936d981747962eae755a81489f62858'),
 ('D1', 'qasper-train:1901.01911:b7e419d2c4e24c40b8ad0fae87036110297d6752')]


def variants(prompt, answer, vocab):
    tail = list(answer[:-1])
    replacement = [((x + 137) % (vocab - 3)) + 3 for x in tail]
    return [('prompt_1', list(prompt)), ('prompt_2', list(prompt)),
            ('teacher_original', list(prompt) + tail),
            ('teacher_repeat', list(prompt) + tail),
            ('teacher_same_length_replaced', list(prompt) + replacement),
            ('teacher_half_length', list(prompt) + tail[:max(1, len(tail)//2)]),
            ('teacher_double_length', list(prompt) + tail + tail),
            ('prompt_after', list(prompt))]


class RouteCaptured(Exception):
    def __init__(self, selected, stats):
        self.selected, self.stats = selected, stats


def reader_type():
    import torch
    from torch.overrides import TorchFunctionMode
    from sparse_reader import SparseEncbankReader

    class OpAudit(TorchFunctionMode):
        def __init__(self, events):
            super().__init__()
            self.events = events

        def __torch_function__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            name = getattr(func, '__name__', str(func))
            if name == 'matmul':
                self.events.append(dict(op='matmul', input_dtypes=[str(x.dtype) for x in args],
                    input_shapes=[list(x.shape) for x in args], output_dtype=str(result.dtype),
                    output_shape=list(result.shape), autocast_enabled=torch.is_autocast_enabled('cuda')))
            elif name == 'masked_fill_':
                mask = args[1]
                self.events.append(dict(op='masked_fill_', shape=list(mask.shape),
                    mask=mask.detach().cpu().tolist(), value=str(args[2])))
            return result

    class RouteReader(SparseEncbankReader):
        def setup_probe(self, prompt_length, probe_rows, instrument=True, stop_at_route=True):
            self.probe_length, self.probe_rows = prompt_length, list(probe_rows)
            self.instrument, self.stop_at_route = instrument, stop_at_route
            self.events, self.tensors = [], {}
            self.in_documents = False

        def _middle_documents(self, *args, **kwargs):
            self.in_documents = True
            try:
                return super()._middle_documents(*args, **kwargs)
            finally:
                self.in_documents = False

        def _segment(self, layer_index, hidden, positions, prefix=None, probe_rows=None):
            result = super()._segment(layer_index, hidden, positions, prefix, probe_rows)
            if self.instrument and not self.in_documents and layer_index < self.m:
                ids = torch.tensor(self.probe_rows, device=hidden.device)
                self.tensors[f'layer{layer_index}.prompt_probe_hidden'] = result[0].index_select(1, ids).detach().cpu()
                self.events.append(dict(op='query_segment', layer=layer_index,
                    input_shape=list(hidden.shape), output_dtype=str(result[0].dtype),
                    prefix_length=0 if prefix is None else prefix.length,
                    query_attention='causal original _segment/_attention',
                    positions_first=int(positions[0, 0]), positions_last=int(positions[0, -1])))
            return result

        def _block_attention_mass(self, q_probe, doc_kv, own_kv, probe_rows, sink_len, lengths, layer_index):
            if self.instrument:
                self.tensors[f'layer{layer_index}.q_probe'] = q_probe.detach().cpu()
                self.tensors[f'layer{layer_index}.own_prompt_k'] = own_kv.k[..., :max(probe_rows)+1, :].detach().cpu()
                self.events.append(dict(op='score_scope', layer=layer_index, probe_rows=list(probe_rows),
                    prompt_length=self.probe_length, own_key_scan_stop=max(probe_rows)+1,
                    actual_own_key_length=own_kv.length, sink_length=sink_len,
                    document_lengths=list(lengths), q_probe_dtype=str(q_probe.dtype),
                    doc_k_dtype=str(doc_kv.k.dtype)))
            with OpAudit(self.events) if self.instrument else nullcontext():
                result = super()._block_attention_mass(q_probe, doc_kv, own_kv, probe_rows,
                    sink_len, lengths, layer_index)
            if self.instrument:
                self.tensors[f'layer{layer_index}.block_mass'] = result.detach().cpu()
            return result

        def _select(self, *args, **kwargs):
            selected, stats = super()._select(*args, **kwargs)
            if self.stop_at_route:
                raise RouteCaptured(selected, stats)
            return selected, stats

        def probe(self, sink, memories, ids, prompt_length, probe_rows, instrument=True):
            self.setup_probe(prompt_length, probe_rows, instrument=instrument)
            try:
                self._execute(sink, memories, self._ids(ids), prompt_length, probe_indices=probe_rows)
            except RouteCaptured as result:
                return dict(route=result.stats, events=self.events), self.tensors
            raise AssertionError('The route-only stop was not reached')

    return RouteReader


def compare_tensors(left, right):
    import torch
    result = {}
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        if a.shape != b.shape:
            result[key] = dict(shape_equal=False)
            continue
        delta = a.float() - b.float()
        result[key] = dict(shape_equal=True, exact=torch.equal(a, b),
            max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()))
    return result


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task-root', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    from slurm_gpu_guard import confined, validate_worker_lease
    root, out = confined(args.task_root), confined(args.out, args.task_root)
    if out.exists():
        raise RuntimeError('Fresh diagnostic output required; preserve previous attempts')
    out.mkdir(parents=True)
    lease = os.environ['SPARSE_GPU_LEASE_PATH']
    admissions = [validate_worker_lease(lease, require_idle=True)]
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(16)
    torch.manual_seed(42)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda *a, **kw: False
    from train_8b_baseline import attach_lora, restore_flat
    from encbank.model import Encbank
    import torch.nn.functional as F
    admissions.append(validate_worker_lease(lease, require_idle=True))
    torch.cuda.set_device(0)
    model_path = root/'models/Qwen3-8B'
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    modules = attach_lora(model, 12, 32, 32, torch.float32)
    encbank = Encbank(model, resume_j=12)
    Reader = reader_type()
    rows = {r['id']: r for r in map(json.loads, (root/'data/qasper_pilot/dev.jsonl').read_text().splitlines())}
    results = []
    metadata = dict(torch=torch.__version__, transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0), seed=42, formal_inference_timing=False,
        formal_inference_memory=False, cases=CASES, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reader_sha256=hashlib.sha256((HERE/'sparse_reader.py').read_bytes()).hexdigest(),
        admissions=admissions, thread_count=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        fixed_writer_memories=True, teacher_original_uses_answer_minus_last=True,
        arithmetic='original unmodified reader; post-selection early stop; TorchFunctionMode metadata only')
    dump(out/'metadata.json', metadata)
    current_arm = None
    for arm, ident in CASES:
        validate_worker_lease(lease, require_idle=False)
        if arm != current_arm:
            checkpoint = root/f'outputs/sparse_encbank_20260911/train/{arm}/step250.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if state['step'] != 250 or state['metadata']['recipe']['arm'] != arm:
                raise RuntimeError('Checkpoint identity differs')
            restore_flat(modules, state['named'])
            del state
            current_arm = arm
        row = rows[ident]
        reader = Reader(encbank, fusion_layer=16, retain_ratio=.5,
                        probe_mode='block' if arm == 'B' else 'dense', gradient_checkpointing=False).eval()
        leaf = out/(arm+'_'+row['document_id'])
        leaf.mkdir()
        with torch.no_grad():
            # Match training: sink writer is outside evaluation autocast.
            sink_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            sink = encbank.write_chunk([sink_id]).detach()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            memories = [encbank.write_chunk(c).detach() for c in row['document_chunks']]
            before = [h.detach().clone() for h in memories]
            observations, tensors = {}, {}
            for label, ids in variants(row['prompt_ids'], row['answer_ids'], model.config.vocab_size):
                validate_worker_lease(lease, require_idle=False)
                record, saved = reader.probe(sink, memories, ids, len(row['prompt_ids']), row['probe_indices'])
                record.update(label=label, query_length=len(ids), prompt_length=len(row['prompt_ids']))
                observations[label], tensors[label] = record, saved
                dump(leaf/(label+'.json'), record)
                torch.save(saved, leaf/(label+'.pt'))
                print(json.dumps(dict(event='probe', arm=arm, id=ident, label=label,
                    selected=record['route']['selected_indices'])), flush=True)
            # Confirm metadata interception does not change the route score.
            plain, _ = reader.probe(sink, memories, row['prompt_ids'], len(row['prompt_ids']), row['probe_indices'], False)
            # Run the full original paths once to validate the early stop result.
            reader.setup_probe(len(row['prompt_ids']), row['probe_indices'], False, False)
            logits, gen_state = reader.prefill(sink, memories, row['prompt_ids'], probe_indices=row['probe_indices'])
            full_gen_route = gen_state.route_stats
            del logits, gen_state
            full_ce = reader.forward_answer(sink, memories, row['prompt_ids'], row['answer_ids'], probe_indices=row['probe_indices'])
            labels = torch.tensor(row['answer_ids'], device='cuda')
            original_ce = float(F.cross_entropy(full_ce['logits'][0].float(), labels))
            full_ce_route = full_ce['route_stats']
            del full_ce
            forced = reader.forward_answer(sink, memories, row['prompt_ids'], row['answer_ids'],
                probe_indices=row['probe_indices'], forced_selection=full_gen_route['selected_indices'])
            forced_ce = float(F.cross_entropy(forced['logits'][0].float(), labels))
            del forced
            rebuilt = [encbank.write_chunk(c).detach() for c in row['document_chunks']]
            writer_diff = [dict(exact=torch.equal(a, b), max_abs=float((a.float()-b.float()).abs().max()))
                           for a, b in zip(memories, rebuilt)]
            memory_unchanged = all(torch.equal(a, b) for a, b in zip(before, memories))
        original = json.loads((root/f'outputs/sparse_encbank_20260911/train/{arm}/eval_step250.json').read_text())
        old = next(x for x in original['records'] if x['id'] == ident)
        pairs = [('prompt_1','prompt_2'), ('prompt_1','prompt_after'), ('teacher_original','teacher_repeat'),
                 ('teacher_original','teacher_same_length_replaced'), ('prompt_1','teacher_original'),
                 ('teacher_original','teacher_half_length'), ('teacher_original','teacher_double_length')]
        comparison = {a+'__vs__'+b: dict(tensors=compare_tensors(tensors[a], tensors[b]),
            route_equal=observations[a]['route']['selected_indices']==observations[b]['route']['selected_indices'],
            scores_max_abs=max(abs(x-y) for x,y in zip(observations[a]['route']['block_scores'],observations[b]['route']['block_scores'])))
            for a,b in pairs}
        summary = dict(arm=arm, id=ident, memory_unchanged=memory_unchanged, writer_rebuild=writer_diff,
            comparisons=comparison, instrumentation_plain_route=plain['route'], full_generation_route=full_gen_route,
            full_ce_route=full_ce_route, original_ce=original_ce, forced_prompt_route_ce=forced_ce,
            saved_generation_route=old['route_stats'], saved_ce_route=old['ce_route_stats'],
            historical_generation_selection_reproduced=full_gen_route['selected_indices']==old['route_stats']['selected_indices'],
            historical_ce_selection_reproduced=full_ce_route['selected_indices']==old['ce_route_stats']['selected_indices'],
            full_vs_probe_gen_scores_equal=full_gen_route['block_scores']==observations['prompt_1']['route']['block_scores'],
            full_vs_probe_ce_scores_equal=full_ce_route['block_scores']==observations['teacher_original']['route']['block_scores'],
            instrumentation_scores_equal=plain['route']['block_scores']==observations['prompt_1']['route']['block_scores'])
        dump(leaf/'summary.json', summary)
        results.append(summary)
        dump(out/'progress.json', dict(complete=False, completed_cases=len(results), results=results))
        del memories, rebuilt, before, sink, reader, tensors
    dump(out/'summary.json', dict(complete=True, completed_cases=len(results), results=results,
        claim_limit='Three fixed cases on the original GPU family; no broad quality or performance claim'))
    print(json.dumps(dict(event='complete', completed_cases=len(results))), flush=True)


if __name__ == '__main__':
    main()
