"""Separate CacheBlend-style entry point; same serving loops/gate/accounting.

No baseline source file is edited. Reader/arm injection is confined to this
standalone process and restored on return, including failures.
"""
from unittest.mock import patch
import argparse
import sys
from pathlib import Path
import torch
import serving_reuse
from cacheblend_serving_reuse import ReusableContextualCacheBlend


def reader_factory(model, j, tokenizer, arm, model_id, adapter_info=None):
    if arm != "cacheblend16" or adapter_info is not None:
        raise ValueError("This driver only admits stock-weight contextual CacheBlend16")
    return ReusableContextualCacheBlend(model, j, tokenizer, model_id, .16)


@torch.no_grad()
def fresh_reference_ids(reader, query_ids, indices, max_new_tokens, force_length):
    """Existing fresh CacheBlend path with the cost harness's fixed-G EOS rule.

    Like generate_explicit this captures sink, selected chunks and query anew.
    The only generation change is suppressing EOS on later tokens when fixed G
    is requested (both paths always suppress EOS on the first token).
    """
    cb, cm = reader.cb, reader.cm
    pieces = [cm._as_ids([cm._sink_prefix_id()])]
    pieces.extend(cm._as_ids(reader.chunks[i]) for i in indices)
    pieces.append(cm._as_ids(query_ids))
    offsets, offset = [], 0
    for piece in pieces:
        offsets.append(offset)
        offset += piece.numel()
    caches = [cb.prefill_chunk_full(piece)[0] for piece in pieces]
    merged = cb.concat_kv_reindex(caches, offsets)
    selection = {}
    logits, _, mixed = cb.read(torch.cat(pieces, dim=1), merged, 1, pieces[-1].numel(), cb.recompute_ratio, selection)
    cache = cb.decode_cache(mixed)
    _, eos = cm._bos_eos(cm.tokenizer)
    first_logits = logits[0, -1].float().clone()
    if eos is not None:
        first_logits[eos] = -torch.inf
    generated = [int(first_logits.argmax())]
    for step in range(1, max_new_tokens):
        next_logits = cb.decode_step(generated[-1], cache, offset + step - 1)[0, -1].float().clone()
        if force_length and eos is not None:
            next_logits[eos] = -torch.inf
        token = int(next_logits.argmax())
        if not force_length and token == eos:
            break
        generated.append(token)
    return generated, selection['selected_positions'], len(pieces)


def persistent_versions(reader):
    files = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in reader.path.iterdir() if p.is_file()}
    versions = None if reader.payloads is None else {
        name: [(id(k), k._version, id(v), v._version) for k, v in item['kv']]
        for name, item in reader.payloads.items()}
    return files, versions


class VerifyingReader(ReusableContextualCacheBlend):
    """GPU-smoke-only fresh reference, outside the adapter's timed intervals."""
    checks = None

    def query_ids(self, query_ids, **kwargs):
        before = persistent_versions(self)
        with patch.object(self.cb, 'prefill_chunk_full', side_effect=AssertionError('A reuse query captured a document/query chunk')):
            actual, stats = super().query_ids(query_ids, **kwargs)
        assert persistent_versions(self) == before, 'Persistent CPU payload or disk store changed during query'
        expected, positions, captures = fresh_reference_ids(self, query_ids, stats['selected_indices'],
            kwargs.get('max_new_tokens', 16), kwargs.get('force_length', False))
        assert actual == expected, 'Reusable CacheBlend token IDs differ from the fresh fixed-G reference'
        assert positions == stats['cacheblend']['selected_positions'], 'Selected recompute positions differ'
        assert persistent_versions(self) == before, 'Fresh reference changed persistent cache'
        check = {'tier': self.tier, 'G': kwargs.get('max_new_tokens', 16), 'generated_ids': expected,
                 'selected_indices': stats['selected_indices'], 'equal_ids': True, 'equal_selected_positions': True,
                 'persistent_cpu_versions_and_disk_metadata_unchanged': True, 'reuse_capture_calls': 0,
                 'extra_reference_sequences': 1, 'extra_reference_prefill_calls': captures,
                 'reference_outside_recorded_query_timing': True}
        stats['fresh_reference'] = check
        stats['instrumentation_diagnostic'] = True
        stats['timing_eligible'] = False
        self.checks.append(check)
        return actual, stats


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # The inherited class still validates its internal stock Encbank constructor
    # against ALL_ARMS. Keep those choices available internally, but reject
    # external attempts to run another arm before model loading/admission.
    preflight = argparse.ArgumentParser(add_help=False)
    preflight.add_argument('--arms', nargs='+')
    preflight.add_argument('--verify-fresh', action='store_true')
    preflight.add_argument('--out', type=Path)
    selected, _ = preflight.parse_known_args(argv)
    if selected.arms is not None and selected.arms != ['cacheblend16']:
        raise ValueError('This standalone driver only runs --arms cacheblend16')
    checks = []
    if selected.verify_fresh:
        argv.remove('--verify-fresh')
    def factory(model, j, tokenizer, arm, model_id, adapter_info=None):
        if arm != 'cacheblend16' or adapter_info is not None:
            raise ValueError('This driver only admits stock-weight contextual CacheBlend16')
        if not selected.verify_fresh:
            return reader_factory(model, j, tokenizer, arm, model_id, adapter_info)
        reader = VerifyingReader(model, j, tokenizer, model_id, .16)
        reader.checks = checks
        return reader
    with patch.object(serving_reuse, "ARMS", ("cacheblend16",)), \
         patch.object(serving_reuse, "ALL_ARMS", serving_reuse.ALL_ARMS + ("cacheblend16",)), \
         patch.object(serving_reuse, "ReusableReader", factory):
        result = serving_reuse.main(argv)
    if selected.verify_fresh:
        serving_reuse.save_json({'complete': True, 'checks': checks, 'verified_queries': len(checks),
            'extra_reference_sequences': len(checks), 'timing_eligible': False,
            'purpose': 'Correctness smoke only; extra fresh references run outside recorded timing'},
            selected.out / 'FRESH_REFERENCE_CHECK.json')
    return result


if __name__ == "__main__":
    raise SystemExit(main())
