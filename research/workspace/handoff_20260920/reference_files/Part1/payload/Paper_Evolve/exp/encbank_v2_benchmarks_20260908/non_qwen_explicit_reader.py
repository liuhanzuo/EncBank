"""Independent five-arm explicit-pack entry; shares no Qwen adapter/guard.

All arms use exactly one fixture, its native template, explicit chunks/query,
the native BOS sink, and the existing benchmark's first-step EOS suppression.
No model loading, scheduler, GPU allocation or benchmark side effect at import.
"""
from __future__ import annotations
import hashlib
import json
from types import SimpleNamespace
import torch
from non_qwen_llama_candidate import DenseLlamaCandidate, Encbank

ARMS = ("pub", "pub_sink", "fix_all", "j0", "fix_none")

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def validate_fixture(fixture, config):
    expected = fixture.get("fixture_sha256")
    if expected != digest({k: v for k, v in fixture.items() if k != "fixture_sha256"}):
        raise ValueError("fixture identity changed")
    if fixture["bos_token_id"] != config.bos_token_id:
        raise ValueError("native BOS mismatch")
    eos = config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    if fixture["eos_token_ids"] != eos or not fixture["suppress_first_eos"]:
        raise ValueError("all arms require recorded native EOS and first-step suppression")
    context, query = fixture["context_ids"], fixture["query_ids"]
    if not query or len(context) != fixture["context_token_count"] or len(query) != fixture["query_token_count"]:
        raise ValueError("explicit boundary count mismatch")
    chunks = [context[i:i + fixture["chunk_size"]] for i in range(0, len(context), fixture["chunk_size"])]
    selected = fixture["ordered_selected_indices"]
    if len(set(selected)) != len(selected) or any(i < 0 or i >= len(chunks) for i in selected):
        raise ValueError("retrieval indices invalid")
    if list(map(len, chunks)) != fixture["context_chunk_lengths"]:
        raise ValueError("chunk boundary mismatch")
    selected_chunks = [chunks[i] for i in selected]
    if list(map(len, selected_chunks)) != fixture["selected_chunk_lengths"]:
        raise ValueError("selected chunk lengths changed")
    selected_ids = [x for c in selected_chunks for x in c]
    pack = [config.bos_token_id] + selected_ids + query
    for name, ids in (("context_ids", context), ("query_ids", query), ("selected_ids", selected_ids), ("pack_ids", pack)):
        if digest(ids) != fixture[name + "_sha256"]:
            raise ValueError(name + " identity changed")
    if len(pack) != fixture["read_pack_tokens"] or len(pack) + fixture["max_new_tokens"] != fixture["conservative_read_plus_cap"]:
        raise ValueError("read budget changed")
    if not fixture["budget_valid"] or fixture["conservative_read_plus_cap"] > config.max_position_embeddings:
        raise ValueError("native read-window limit exceeded")
    return selected_chunks, query, pack

def pick_token(logits, eos_ids, step):
    """Shared benchmark convention; return raw logits separately for auditing."""
    choice = logits[0, -1].float().clone()
    if step == 0:
        choice[eos_ids] = float("-inf")
    return int(choice.argmax())

class NonQwenExplicitReader:
    def __init__(self, model, arm, *, j=8, model_binding, tokenizer=None):
        if arm not in ARMS:
            raise ValueError("unknown independent arm")
        self.arm, self.model, self.j = arm, model, j
        self.tokenizer = tokenizer
        # Apply the narrow dense-Llama/default-RoPE guard to every arm.
        checked = DenseLlamaCandidate(model, 0 if arm == "j0" else j,
                                      model_binding=model_binding, lower_layers=[] if arm == "fix_none" else None)
        if arm in ("fix_all", "fix_none"):
            self.reader = checked
        else:
            tokens = tokenizer or SimpleNamespace(bos_token_id=model.config.bos_token_id,
                                                  eos_token_id=model.config.eos_token_id)
            self.reader = Encbank(model, 0 if arm == "j0" else j, tokenizer=tokens)
            self.reader.write_sink = arm == "pub_sink"

    @torch.no_grad()
    def generate_fixture(self, fixture, *, capture_logits=False):
        selected, query, pack = validate_fixture(fixture, self.model.config)
        reader = self.reader
        try:
            if isinstance(reader, DenseLlamaCandidate):
                # Current accuracy path writes only selected chunks per query;
                # this is not a write-once service-cost or cache-reuse claim.
                document = reader.write_document(selected)
                sink, hidden = reader.prepare_read(document, list(range(len(selected))))
            else:
                saved = reader.write_sink
                try:
                    reader.write_sink = False
                    sink = reader.write_chunk([fixture["bos_token_id"]])
                finally:
                    reader.write_sink = saved
                hidden = [reader.write_chunk(c) for c in selected]
            qh, bottom, qpos = reader.write_prefill(query)
            logits, upper, ppos = reader.read_prefill(sink, hidden, qh)
            assert ppos == len(pack)
            output = {"arm": self.arm, "effective_j": reader.resume_j, "ids": [], "decoder_forwards": 0,
                      "stopped_on_eos": False, "fixture_sha256": fixture["fixture_sha256"],
                      "pack_ids_sha256": fixture["pack_ids_sha256"], "query_ids_sha256": fixture["query_ids_sha256"],
                      "ordered_selected_indices": fixture["ordered_selected_indices"], "actual_read_pack_tokens": ppos,
                      "max_new_tokens": fixture["max_new_tokens"], "eos_token_ids": fixture["eos_token_ids"],
                      "suppress_first_eos": True, "write_protocol": "selected chunks captured per query",
                      "first_raw_argmax": int(logits[0, -1].argmax())}
            if capture_logits:
                output["step_logits"] = []
            for step in range(fixture["max_new_tokens"]):
                if capture_logits:
                    output["step_logits"].append(logits.detach().cpu().clone())
                token = pick_token(logits, fixture["eos_token_ids"], step)
                output["ids"].append(token)
                if token in fixture["eos_token_ids"]:
                    output["stopped_on_eos"] = True
                    break
                if step + 1 < fixture["max_new_tokens"]:
                    logits = reader.decode_step(token, bottom, upper, qpos, ppos)
                    qpos += 1
                    ppos += 1
                    output["decoder_forwards"] += 1
            output["first_eos_suppressed"] = output["first_raw_argmax"] in fixture["eos_token_ids"]
            output["generated_tokens"] = len(output["ids"])
            if self.tokenizer is not None:
                output["decoded_output"] = self.tokenizer.decode(output["ids"], skip_special_tokens=True).strip()
            return output
        finally:
            if isinstance(reader, DenseLlamaCandidate):
                reader.clear_read()

@torch.no_grad()
def native_reference(model, fixture, *, capture_logits=False):
    """Stock full-pack reference using the identical shared EOS convention."""
    _, _, pack = validate_fixture(fixture, model.config)
    device = next(model.parameters()).device
    out = model(input_ids=torch.tensor([pack], device=device), use_cache=True)
    result = {"ids": [], "decoder_forwards": 0, "reference_sequences": 1,
              "fixture_sha256": fixture["fixture_sha256"], "suppress_first_eos": True}
    if capture_logits:
        result["step_logits"] = []
    for step in range(fixture["max_new_tokens"]):
        if capture_logits:
            result["step_logits"].append(out.logits[:, -1:].detach().cpu().clone())
        token = pick_token(out.logits, fixture["eos_token_ids"], step)
        result["ids"].append(token)
        if token in fixture["eos_token_ids"]:
            break
        if step + 1 < fixture["max_new_tokens"]:
            out = model(input_ids=torch.tensor([[token]], device=device), past_key_values=out.past_key_values, use_cache=True)
            result["decoder_forwards"] += 1
    return result
