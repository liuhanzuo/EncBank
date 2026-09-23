"""Shared SFT preparation and generation-only QA evaluation for Encbank wrappers.

This module never loads a model and reports no hardware timing. The writer sees
only independently selected document chunks. Answers enter a separate, optional
teacher-forced CE pass, never retrieval, writing, or free generation.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import string
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


PROMPT_INSTRUCTION = "Answer the question using the provided document memory. Give only the answer."


@dataclass(frozen=True)
class PreparedExample:
    id: str
    document_id: str
    source: str
    split: str
    question: str
    document_chunks: tuple[tuple[int, ...], ...]
    selected_chunk_indices: tuple[int, ...]
    original_context_tokens: int
    prompt_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    references: tuple[str, ...]
    answer_truncated: bool = False


class PreparedExamples(list):
    """A normal list with explicit filtering/protocol metadata attached."""

    def __init__(self, values=(), *, skipped=(), preparation_summary=None):
        super().__init__(values)
        self.skipped = list(skipped)
        self.preparation_summary = dict(preparation_summary or {})


def _ids(value) -> tuple[int, ...]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected one token sequence")
        value = value[0]
    return tuple(int(item) for item in value)


def build_prompt_ids(tokenizer, question: str) -> tuple[int, ...]:
    """Use exactly the same non-thinking chat prompt in SFT and generation."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    messages = [{"role": "user", "content": f"{PROMPT_INSTRUCTION}\n\nQuestion: {question}"}]
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    ids = _ids(result)
    if not ids:
        raise ValueError("The chat template produced an empty prompt")
    return ids


def eos_token_ids(tokenizer, wrapper=None) -> set[int]:
    """Include model EOS lists and Qwen's assistant-turn terminator."""
    result: set[int] = set()

    def add(value):
        if value is None:
            return
        values = value if isinstance(value, (tuple, list, set)) else [value]
        result.update(int(item) for item in values if item is not None and int(item) >= 0)

    add(getattr(tokenizer, "eos_token_id", None))
    for owner in (wrapper, getattr(wrapper, "model", None)):
        add(getattr(getattr(owner, "generation_config", None), "eos_token_id", None))
        add(getattr(getattr(owner, "config", None), "eos_token_id", None))
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        token_id = convert("<|im_end|>")
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            add(token_id)
    return result


def _answer_eos(tokenizer) -> int | None:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        token_id = convert("<|im_end|>")
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            return int(token_id)
    values = getattr(tokenizer, "eos_token_id", None)
    if isinstance(values, (tuple, list)):
        return int(values[0]) if values else None
    return int(values) if values is not None else None


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def _terms(text: str) -> Counter:
    return Counter(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


class _BM25:
    """Chunk BM25 with fixed k1/b; query input contains no reference answers."""

    def __init__(self, chunks: Sequence[Sequence[int]], tokenizer):
        self.counts = [_terms(tokenizer.decode(list(chunk), skip_special_tokens=True)) for chunk in chunks]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.document_frequency = Counter(term for count in self.counts for term in count)

    def select(self, question: str, max_chunks: int) -> tuple[int, ...]:
        if len(self.counts) <= max_chunks:
            return tuple(range(len(self.counts)))
        query = _terms(question)
        count = len(self.counts)
        scores = []
        for index, terms in enumerate(self.counts):
            score = 0.0
            length_factor = 1.2 * (1.0 - 0.75 + 0.75 * self.lengths[index] / max(self.average_length, 1.0))
            for term in query:
                frequency = terms.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log1p((count - df + 0.5) / (df + 0.5))
                score += idf * frequency * 2.2 / (frequency + length_factor)
            scores.append((score, index))
        chosen = sorted(scores, key=lambda item: (-item[0], item[1]))[:max_chunks]
        return tuple(sorted(index for _, index in chosen))


def prepare_examples(
    rows: Iterable[Mapping[str, Any]], tokenizer, *, chunk_size: int = 512,
    max_chunks: int = 7, max_question_tokens: int = 256, max_answer_tokens: int = 128,
    max_context_tokens: int | None = None, max_examples: int | None = None, seed: int = 42,
) -> PreparedExamples:
    """Prepare canonical JSONL rows without answer-dependent document selection.

    Full contexts are tokenized and chunked before question-only BM25 retrieval.
    A context length limit, if explicitly provided, filters rather than truncates.
    Reference strings stay complete; only teacher-forcing target IDs are capped.
    Too-long questions are explicitly filtered instead of truncating a chat template.
    """
    if min(chunk_size, max_chunks, max_question_tokens, max_answer_tokens) < 1:
        raise ValueError("All token/chunk limits must be positive")
    if max_context_tokens is not None and max_context_tokens < 1:
        raise ValueError("max_context_tokens must be positive when supplied")
    if max_examples is not None and max_examples < 0:
        raise ValueError("max_examples cannot be negative")
    values = list(rows)
    document_cache: dict[str, tuple[str, tuple[tuple[int, ...], ...], _BM25, int]] = {}
    seen_ids: set[str] = set()
    result, skipped = [], []
    answer_eos = _answer_eos(tokenizer)
    for row in values:
        required = ("id", "document_id", "context", "question", "answer", "source", "split")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Missing canonical fields: {missing}")
        example_id, document_id = str(row["id"]), str(row["document_id"])
        if not example_id or not document_id:
            raise ValueError("id and document_id must be nonempty")
        if example_id in seen_ids:
            raise ValueError(f"Duplicate example id: {example_id}")
        seen_ids.add(example_id)
        context, question = row["context"], row["question"]
        if not isinstance(context, str) or not isinstance(question, str):
            raise ValueError(f"context/question must be strings: {example_id}")
        references = row["answer"]
        if isinstance(references, str):
            references = (references,)
        elif isinstance(references, (tuple, list)) and references and all(isinstance(x, str) for x in references):
            references = tuple(references)
        else:
            raise ValueError(f"answer must be a string or nonempty list of strings: {example_id}")
        aliases = row.get("answers", [])
        if aliases is None:
            aliases = []
        if not isinstance(aliases, (tuple, list)) or not all(isinstance(x, str) for x in aliases):
            raise ValueError(f"answers must be a list of reference strings: {example_id}")
        # The primary answer remains first for SFT/CE; aliases affect scoring only.
        references = tuple(dict.fromkeys((*references, *aliases)))
        if not question.strip():
            skipped.append({"id": example_id, "reason": "empty_question"})
            continue
        question_ids = _ids(tokenizer.encode(question, add_special_tokens=False))
        if len(question_ids) > max_question_tokens:
            skipped.append({"id": example_id, "reason": "question_too_long", "question_tokens": len(question_ids)})
            continue
        if document_id in document_cache:
            cached_context, chunks, index, context_length = document_cache[document_id]
            if context != cached_context:
                raise ValueError(f"Conflicting contexts for document_id {document_id}")
        else:
            context_ids = _ids(tokenizer.encode(context, add_special_tokens=False))
            context_length = len(context_ids)
            chunks = tuple(context_ids[start:start + chunk_size] for start in range(0, context_length, chunk_size))
            index = _BM25(chunks, tokenizer)
            document_cache[document_id] = (context, chunks, index, context_length)
        if not context_length:
            skipped.append({"id": example_id, "reason": "empty_document"})
            continue
        if max_context_tokens is not None and context_length > max_context_tokens:
            skipped.append({"id": example_id, "reason": "context_too_long", "context_tokens": context_length})
            continue
        selected = index.select(question, max_chunks)
        raw_answer_ids = _ids(tokenizer.encode(references[0], add_special_tokens=False))
        # EOS is part of the target budget. Never teach EOS for a truncated answer.
        complete_target = raw_answer_ids + ((answer_eos,) if answer_eos is not None else ())
        answer_truncated = len(complete_target) > max_answer_tokens
        answer_ids = raw_answer_ids[:max_answer_tokens] if answer_truncated else complete_target
        if not answer_ids:
            skipped.append({"id": example_id, "reason": "empty_answer_without_eos"})
            continue
        result.append(PreparedExample(
            id=example_id, document_id=document_id, source=str(row["source"]), split=str(row["split"]),
            question=question, document_chunks=tuple(chunks[i] for i in selected),
            selected_chunk_indices=selected, original_context_tokens=context_length,
            prompt_ids=build_prompt_ids(tokenizer, question), answer_ids=answer_ids,
            references=tuple(references), answer_truncated=answer_truncated,
        ))
    if max_examples is not None and len(result) > max_examples:
        ranked = sorted(result, key=lambda item: (hashlib.sha256(f"{seed}:{item.id}".encode()).hexdigest(), item.id))
        keep = {item.id for item in ranked[:max_examples]}
        for item in result:
            if item.id not in keep:
                skipped.append({"id": item.id, "reason": "fixed_subsample"})
        result = [item for item in result if item.id in keep]
    summary = {
        "input_examples": len(values), "prepared_examples": len(result), "skipped_examples": len(skipped),
        "skip_reasons": dict(Counter(item["reason"] for item in skipped)), "seed": seed,
        "chunk_size": chunk_size, "max_chunks": max_chunks, "max_question_tokens": max_question_tokens,
        "max_answer_tokens": max_answer_tokens, "max_context_tokens": max_context_tokens,
        "max_examples": max_examples, "retrieval": "question-only-BM25-k1=1.2-b=0.75-source-order",
        "context_policy": "full-document-chunking-no-head-truncation", "enable_thinking": False,
        "answer_truncated_examples": sum(item.answer_truncated for item in result),
    }
    return PreparedExamples(result, skipped=skipped, preparation_summary=summary)


def _field(example, name):
    return example[name] if isinstance(example, Mapping) else getattr(example, name)


def _memory_bytes(memory) -> int:
    hidden = memory if torch.is_tensor(memory) else getattr(memory, "hidden", None)
    return int(hidden.numel() * hidden.element_size()) if torch.is_tensor(hidden) else 0


def _last_logits(logits) -> torch.Tensor:
    if not torch.is_tensor(logits) or logits.ndim < 1:
        raise ValueError("Wrapper logits must be a tensor")
    values = logits.reshape(-1, logits.shape[-1])[-1]
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite generation logits")
    return values


def _summary(records, cache_stats, protocol) -> dict:
    def scores(items):
        ce_tokens = sum(item.get("answer_ce_tokens", 0) for item in items)
        return {
            "examples": len(items),
            "exact_match": sum(item["exact_match"] for item in items) / len(items) if items else None,
            "token_f1": sum(item["token_f1"] for item in items) / len(items) if items else None,
            "answer_ce": sum(item.get("answer_ce_sum", 0.0) for item in items) / ce_tokens if ce_tokens else None,
            "answer_ce_tokens": ce_tokens,
        }
    result = scores(records)
    result.update({"score_scale": "0-to-1", "by_source": {}, "cache": dict(cache_stats), "protocol": protocol})
    for source in sorted({item["source"] for item in records}):
        result["by_source"][source] = scores([item for item in records if item["source"] == source])
    return result


def evaluate_examples(
    wrapper, tokenizer, prepared_examples: Sequence, max_new_tokens: int = 64, *,
    answer_ce: bool = False, output_path: str | Path | None = None,
    max_cached_chunks: int = 128,
) -> dict:
    """Greedy natural-EOS generation, optional independent answer CE, per-item records.

    ``wrapper`` implements encode_chunk(ids, cache_device="cpu"),
    start_query(prompt_ids, memories), decode_token(token, state), and optionally
    read_logits(ids, memories). Each state exposes its current ``logits``.
    The CPU LRU is local to this call, so it cannot survive a writer update.
    """
    if max_new_tokens < 1 or max_cached_chunks < 1:
        raise ValueError("Generation and cache limits must be positive")
    if answer_ce and not callable(getattr(wrapper, "read_logits", None)):
        raise ValueError("answer_ce requires wrapper.read_logits")
    examples = list(prepared_examples)
    identifiers = [str(_field(item, "id")) for item in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Evaluation IDs must be unique")
    stops = eos_token_ids(tokenizer, wrapper)
    cache: OrderedDict[tuple[str, int], tuple[tuple[int, ...], Any]] = OrderedDict()
    seen_chunk_content: dict[tuple[str, int], tuple[int, ...]] = {}
    records = []
    stats = {"chunk_writes": 0, "chunk_hits": 0, "evictions": 0, "peak_cpu_payload_bytes": 0}
    protocol = {
        "decoding": "greedy", "max_new_tokens": max_new_tokens, "eos_token_ids": sorted(stops),
        "enable_thinking": False, "answer_ce": answer_ce, "ce_reference_index": 0,
        "ce_targets": "prepared-answer-ids-including-eos-only-when-not-truncated",
        "cache_key": "document_id+original_chunk_index-with-content-validation",
        "max_cached_chunks": max_cached_chunks, "hardware_timing": "not-collected",
        "preparation": getattr(prepared_examples, "preparation_summary", None),
    }
    output = Path(output_path) if output_path is not None else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("w", encoding="utf-8") if output is not None else None
    mode_owner = wrapper if callable(getattr(wrapper, "eval", None)) else getattr(wrapper, "model", None)
    previous_training = getattr(mode_owner, "training", None)
    if callable(getattr(mode_owner, "eval", None)):
        mode_owner.eval()
    try:
        with torch.no_grad():
            for example in examples:
                document_id = str(_field(example, "document_id"))
                chunks = _field(example, "document_chunks")
                indices = _field(example, "selected_chunk_indices")
                if len(chunks) != len(indices) or not chunks:
                    raise ValueError("Selected chunks/indices must be nonempty and aligned")
                memories = []
                for original_index, chunk in zip(indices, chunks):
                    key, content = (document_id, int(original_index)), _ids(chunk)
                    if key in seen_chunk_content and seen_chunk_content[key] != content:
                        raise ValueError(f"Conflicting document chunk content: {key}")
                    seen_chunk_content[key] = content
                    if key in cache:
                        _, memory = cache.pop(key)
                        stats["chunk_hits"] += 1
                    else:
                        memory = wrapper.encode_chunk(content, cache_device="cpu")
                        stats["chunk_writes"] += 1
                    cache[key] = (content, memory)
                    memories.append(memory)
                    while len(cache) > max_cached_chunks:
                        cache.popitem(last=False)
                        stats["evictions"] += 1
                    stats["peak_cpu_payload_bytes"] = max(
                        stats["peak_cpu_payload_bytes"], sum(_memory_bytes(value[1]) for value in cache.values()),
                    )
                prompt = _ids(_field(example, "prompt_ids"))
                if not prompt:
                    raise ValueError("Prepared prompt cannot be empty")
                state = wrapper.start_query(prompt, memories)
                generated = []
                finish_reason = "max_new_tokens"
                for step in range(max_new_tokens):
                    token = int(_last_logits(state.logits).argmax())
                    generated.append(token)
                    if token in stops:
                        finish_reason = "eos"
                        break
                    if step + 1 < max_new_tokens:
                        logits = wrapper.decode_token(token, state)
                        # The protocol permits wrappers that also return current logits.
                        if logits is not None:
                            state.logits = logits
                del state
                visible_ids = generated[:-1] if finish_reason == "eos" else generated
                prediction = tokenizer.decode(visible_ids, skip_special_tokens=True).strip()
                references = tuple(str(value) for value in _field(example, "references"))
                if not references:
                    raise ValueError("Every example needs at least one scoring reference")
                record = {
                    "id": str(_field(example, "id")), "document_id": document_id,
                    "source": str(_field(example, "source")), "split": str(_field(example, "split")),
                    "question": str(_field(example, "question")), "references": list(references),
                    "prediction": prediction, "generated_ids": generated, "generated_tokens": len(generated),
                    "finish_reason": finish_reason, "exact_match": max(exact_match(prediction, ref) for ref in references),
                    "token_f1": max(token_f1(prediction, ref) for ref in references),
                    "selected_chunk_indices": list(map(int, indices)), "selected_context_tokens": sum(len(x) for x in chunks),
                    "original_context_tokens": int(_field(example, "original_context_tokens")),
                    "prompt_tokens": len(prompt), "answer_truncated": bool(_field(example, "answer_truncated")),
                }
                if answer_ce:
                    targets = _ids(_field(example, "answer_ids"))
                    if not targets:
                        raise ValueError("Teacher-forced answer targets cannot be empty")
                    full_input = prompt + targets[:-1]
                    logits = wrapper.read_logits(full_input, memories)
                    positions = logits.reshape(-1, logits.shape[-1])[-len(targets):].float()
                    if positions.shape[0] != len(targets) or not bool(torch.isfinite(positions).all()):
                        raise ValueError("Invalid answer-position CE logits")
                    labels = torch.tensor(targets, dtype=torch.long, device=positions.device)
                    ce_sum = float(F.cross_entropy(positions, labels, reduction="sum"))
                    record.update(answer_ce=ce_sum / len(targets), answer_ce_sum=ce_sum, answer_ce_tokens=len(targets))
                records.append(record)
                if stream is not None:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
    finally:
        if stream is not None:
            stream.close()
        if previous_training is not None and callable(getattr(mode_owner, "train", None)):
            mode_owner.train(previous_training)
    result = {"summary": _summary(records, stats, protocol), "records": records}
    if output is not None:
        summary_path = output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
