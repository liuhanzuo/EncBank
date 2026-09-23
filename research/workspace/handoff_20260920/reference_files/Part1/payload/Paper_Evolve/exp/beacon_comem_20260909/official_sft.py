"""CPU-only preparation of full-conversation Activation-Beacon SFT inputs.

Consumes joined canonical conversations from prepare_official_sft_data, with
both original_messages and document-free messages. This is a Qwen3/CoMem
adaptation, not the original Beacon trainer or its separate RedPajama LM branch.
No document retrieval, token truncation, model loading, or torch imports occur.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
from typing import Iterable, Mapping

IGNORE_INDEX = -100
DOCUMENT_MARKER = "[Document provided in memory.]"
END_TEXT = "<|im_end|>"


class ConversationFiltered(ValueError):
    def __init__(self, reason, **details):
        self.reason, self.details = reason, details
        super().__init__(f"{reason}: {details}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_ids(value):
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (tuple, list)):
        require(len(value) == 1, "Only one conversation per tokenization call")
        value = value[0]
    result = tuple(value)
    require(all(type(v) is int and v >= 0 for v in result), "Invalid tokenizer IDs")
    return result


def render(tokenizer, messages, *, generation=False):
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=generation, enable_thinking=False)
    require(isinstance(text, str) and bool(text), "Native template produced no text")
    return text


def encode(tokenizer, text):
    return token_ids(tokenizer.encode(text, add_special_tokens=False))


def normalize_messages(value):
    require(isinstance(value, (list, tuple)) and len(value) >= 2, "Need complete user/assistant messages")
    messages = []
    for item in value:
        require(isinstance(item, Mapping), "Message must be a mapping")
        require(set(item).issubset({"role", "content"}), "Tool/reasoning message fields are outside this non-thinking protocol")
        role, content = item.get("role"), item.get("content")
        require(role in {"system", "user", "assistant"} and isinstance(content, str), "Unsupported message role/content")
        require(bool(content.strip()), "Empty message content")
        # These are structural chat controls, not literal prose in this template.
        require(not any(s in content for s in ("<|im_start|>", END_TEXT, "<think>", "</think>")),
                "Explicit chat/reasoning control text needs a separate protocol")
        messages.append({"role": role, "content": content})
    start = int(messages[0]["role"] == "system")
    require((len(messages)-start) % 2 == 0, "Conversation must end with assistant")
    for i, message in enumerate(messages[start:]):
        require(message["role"] == ("user" if i % 2 == 0 else "assistant"), "Messages must alternate user/assistant")
    return messages


def validate_document_removal(conversation, messages, original):
    context = conversation.get("context")
    require(isinstance(context, str) and bool(context.strip()), "Need exact source context")
    require(len(messages) == len(original), "Original/adapted conversation message counts differ")
    first_user = int(messages[0]["role"] == "system")
    require(original[first_user]["role"] == "user", "Missing original first user")
    for index, (before, after) in enumerate(zip(original, messages)):
        require(before["role"] == after["role"], "Adaptation changed message roles")
        if index != first_user:
            require(before == after, "Only the first user document span may change")
    raw, adapted = original[first_user]["content"], messages[first_user]["content"]
    prefix = conversation.get("original_first_user_prefix")
    suffix = conversation.get("original_first_user_suffix")
    if prefix is not None or suffix is not None:
        require(isinstance(prefix, str) and isinstance(suffix, str), "Both exact document-span boundaries are required")
        require(raw == prefix+context+suffix and adapted == prefix+DOCUMENT_MARKER+suffix,
                "Exact document span/prefix/suffix reconstruction failed")
    else:
        require(raw.count(context) == 1 and raw.replace(context, DOCUMENT_MARKER, 1) == adapted,
                "Cannot verify exact single document-span replacement")
    return context


@dataclass(frozen=True)
class PreparedAssistantTurn:
    id: str
    assistant_turn_index: int
    message_index: int
    prompt_ids: tuple[int, ...]
    answer: str
    references: tuple[str, ...]
    target_indices: tuple[int, ...]
    training_target_ids: tuple[int, ...]
    target_token_count: int
    conversation_token_weight: float


@dataclass(frozen=True)
class PreparedConversation:
    id: str
    conversation_id: str
    document_id: str
    source: str
    source_file: str
    source_index: object
    split: str
    parser: object
    raw_record_reference: object
    context_sha256: str
    document_chunks: tuple[tuple[int, ...], ...]
    query_ids: tuple[int, ...]
    label_ids: tuple[int, ...]
    target_indices: tuple[int, ...]
    turns: tuple[PreparedAssistantTurn, ...]
    document_token_count: int
    original_template_token_count: int
    reader_template_token_count: int
    raw_uncompressed_read_pack_tokens: int
    target_token_count: int
    assistant_turn_count: int
    protocol: dict

    @property
    def labels(self):
        """Alias consumed by official_sft_objective.conversation_loss."""
        return self.label_ids

    def to_dict(self):
        return asdict(self)


def assistant_spans(tokenizer, messages):
    """Native-render verified character/token spans, never substring-guess answers.

Unique placeholders identify repeated answers independently. Substituting actual
contents must reconstruct the exact native rendering. The fast tokenizer's
offsets must place every supervised token wholly inside one content/end span.
Ambiguous boundaries are rejected instead of masking a user/header token.
"""
    full_text = render(tokenizer, messages)
    assistants = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    marker_messages = [dict(m) for m in messages]
    markers = []
    for turn, index in enumerate(assistants):
        marker = f"OFFICIAL_ASSISTANT_TARGET_{turn}_{hashlib.sha256(full_text.encode()).hexdigest()}"
        require(marker not in full_text, "Placeholder collision")
        markers.append(marker)
        marker_messages[index]["content"] = marker
    marked = render(tokenizer, marker_messages)
    reconstructed, cursor, spans = "", 0, []
    for marker, index in zip(markers, assistants):
        require(marked.count(marker) == 1, "Native template changed assistant placeholder")
        start = marked.index(marker, cursor)
        reconstructed += marked[cursor:start]
        # Qwen3 strips leading newlines from the final assistant's content.
        # Reject such transformations to preserve the complete target literally.
        answer = messages[index]["content"]
        content_start = len(reconstructed)
        reconstructed += answer
        suffix = start+len(marker)
        require(marked.startswith(END_TEXT, suffix), "Native assistant ending is not ChatML im_end")
        spans.append((content_start, len(reconstructed)+len(END_TEXT), index))
        cursor = suffix
    reconstructed += marked[cursor:]
    require(reconstructed == full_text, "Native template transformed answer content; no silent target normalization")
    tokenized = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = token_ids(tokenized)
    offsets = tokenized.get("offset_mapping")
    require(offsets is not None and len(offsets) == len(ids), "Fast tokenizer offsets are required")
    offsets = [tuple(map(int, x)) for x in offsets]
    end_ids = encode(tokenizer, END_TEXT)
    require(len(end_ids) == 1, "ChatML ending must be one native token")
    eos = tokenizer.eos_token_id
    require(end_ids[0] in (eos if isinstance(eos, (list, tuple)) else [eos]), "Template ending differs from native EOS")
    groups, occupied = [], set()
    for start, end, message_index in spans:
        selected = []
        for i, (left, right) in enumerate(offsets):
            if right > start and left < end:
                require(start <= left < right <= end, "Tokenizer token crosses an assistant-label boundary")
                require(i not in occupied, "Overlapping assistant targets")
                selected.append(i); occupied.add(i)
        require(bool(selected) and selected == list(range(selected[0], selected[-1]+1)), "Noncontiguous assistant target span")
        require(offsets[selected[0]][0] == start and offsets[selected[-1]][1] == end,
                "Tokenizer skipped assistant target characters")
        require(ids[selected[-1]] == end_ids[0], "Missing supervised native turn ending")
        groups.append((message_index, tuple(selected)))
    require(token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                    enable_thinking=False)) == ids,
            "Rendered-text and native template tokenization disagree")
    return ids, groups


def prepare_conversation(conversation, tokenizer, *, chunk_size=512, min_length=7200,
                         max_length=20000, max_reader_pack_tokens=None):
    require(chunk_size == 512, "This common-input protocol fixes original chunks at 512 tokens")
    require(type(min_length) is int and type(max_length) is int and 0 <= min_length <= max_length,
            "Invalid full-template length bounds")
    require(isinstance(conversation, Mapping), "Canonical conversation must be a mapping")
    cid = conversation.get("conversation_id", conversation.get("id"))
    require(isinstance(cid, str) and bool(cid), "Need conversation identity")
    for key in ("document_id", "source", "split"):
        require(isinstance(conversation.get(key), str) and bool(conversation[key]), f"Missing {key}")
    require("messages" in conversation and "original_messages" in conversation,
            "Need original/adapted full chat; raw LM text requires a separate objective")
    assistant_contents = [m.get("content") for m in conversation["messages"]
                          if isinstance(m, Mapping) and m.get("role") == "assistant"]
    empty_turns = [i for i, content in enumerate(assistant_contents)
                   if isinstance(content, str) and not content.strip()]
    if empty_turns:
        raise ConversationFiltered("empty_assistant_target", conversation_id=cid,
            assistant_turn_indices=empty_turns, assistant_turn_count=len(assistant_contents),
            policy="exclude entire conversation; retain raw/canonical data; never silently drop a turn")
    messages = normalize_messages(conversation["messages"])
    original = normalize_messages(conversation["original_messages"])
    context = validate_document_removal(conversation, messages, original)
    original_ids = encode(tokenizer, render(tokenizer, original))
    if not min_length <= len(original_ids) <= max_length:
        raise ConversationFiltered("full_original_template_length", conversation_id=cid,
            original_template_tokens=len(original_ids), minimum=min_length, maximum=max_length)
    context_ids = encode(tokenizer, context)
    require(bool(context_ids), "Source document tokenized to empty")
    full_ids, groups = assistant_spans(tokenizer, messages)
    require(len(full_ids) >= 2, "Reader chat is too short for next-token targets")
    query_ids = full_ids[:-1]
    raw_pack = 1+len(context_ids)+len(query_ids)
    if max_reader_pack_tokens is not None and raw_pack > max_reader_pack_tokens:
        raise ConversationFiltered("uncompressed_reader_pack_length", conversation_id=cid,
            raw_uncompressed_read_pack_tokens=raw_pack, maximum=max_reader_pack_tokens)
    labels = [IGNORE_INDEX]*len(query_ids)
    for _, indices in groups:
        for position in indices:
            require(position > 0, "No preceding causal position for assistant target")
            labels[position-1] = full_ids[position]
    target_indices = tuple(i for i, token in enumerate(labels) if token != IGNORE_INDEX)
    require(bool(target_indices), "Conversation has no valid assistant targets")
    supplied_turns = conversation.get("turns")
    if supplied_turns is not None:
        require(len(supplied_turns) == len(groups), "Canonical turn count differs")
    if conversation.get("assistant_turn_count") is not None:
        require(conversation["assistant_turn_count"] == len(groups), "Declared assistant turn count differs")
    turns = []
    for index, (message_index, positions) in enumerate(groups):
        metadata = supplied_turns[index] if supplied_turns is not None else {}
        require(isinstance(metadata, Mapping), "Canonical turn metadata must be a mapping")
        require(metadata.get("assistant_turn_index", index) == index, "Noncontiguous assistant turn indices")
        answer = messages[message_index]["content"]
        require(metadata.get("answer", answer) == answer, "Canonical turn answer differs from messages")
        if "messages" in metadata:
            require(metadata["messages"] == messages[:message_index], "Turn prompt omits history or includes current/future target")
        prompt = encode(tokenizer, render(tokenizer, messages[:message_index], generation=True))
        refs = tuple(metadata.get("answers", [answer]))
        require(bool(refs) and all(isinstance(r, str) for r in refs), "Invalid turn references")
        turns.append(PreparedAssistantTurn(str(metadata.get("id", f"{cid}:assistant{index}")), index,
            message_index, prompt, answer, refs, tuple(p-1 for p in positions),
            tuple(full_ids[p] for p in positions), len(positions), len(positions)/len(target_indices)))
    return PreparedConversation(cid, cid, conversation["document_id"], conversation["source"],
        str(conversation.get("source_file", "")), conversation.get("source_index"), conversation["split"],
        conversation.get("parser"), conversation.get("raw_record_reference"),
        hashlib.sha256(context.encode("utf-8")).hexdigest(),
        tuple(context_ids[i:i+512] for i in range(0, len(context_ids), 512)), query_ids, tuple(labels),
        target_indices, tuple(turns), len(context_ids), len(original_ids), len(full_ids), raw_pack,
        len(target_indices), len(groups), {"format": "official-full-conversation-qwen3-comem-v1",
            "template": "native Qwen3 ChatML; enable_thinking=False", "chunk_size": 512,
            "selector": "all_original_chunks_in_order", "document_marker": DOCUMENT_MARKER,
            "truncated": False, "min_original_template_tokens": min_length,
            "max_original_template_tokens": max_length, "max_uncompressed_reader_pack_tokens": max_reader_pack_tokens,
            "objective": "CE sum over every assistant content/end token divided by this conversation target-token count",
            "conversation_weight": "one unit per conversation, not one unit per expanded turn",
            "template_boundary_policy": "reject transformed targets or ambiguous token boundaries",
            "writer_input": "source context only; no chat histories/questions/assistant targets",
            "lower_query": "full teacher-forced document-free causal conversation; labels shifted once",
            "dev_prompt": "each native generation prompt retains only its legitimate prior history",
            "official_pretraining_LM_branch": "not included"})


class PreparedConversations(list):
    def __init__(self, values=(), *, preparation_summary=None, skipped=()):
        super().__init__(values)
        self.preparation_summary = dict(preparation_summary or {})
        self.skipped = list(skipped)


def iter_prepared_conversations(conversations: Iterable[Mapping], tokenizer, *, report=None, **kwargs):
    """Streaming API for parser's iter_canonical_conversations(processed_dir,...).

    Explicit length/empty-target conversation filters are skipped. Malformed
    canonical/template inputs raise, so loss of a source/assistant round cannot
    silently pass preparation. Consume the iterator before using final counts.
    """
    report = {} if report is None else report
    report.update(input_conversations=0, accepted_conversations=0, accepted_assistant_turns=0,
                  accepted_target_tokens=0, accepted_original_template_tokens=0,
                  accepted_document_tokens=0, filtered_conversations=0,
                  filtered_by_reason={}, skipped=[], complete=False)
    seen = set()
    for conversation in conversations:
        cid = conversation.get("conversation_id", conversation.get("id"))
        require(cid not in seen, "Duplicate canonical conversation identity")
        seen.add(cid); report["input_conversations"] += 1
        try:
            prepared = prepare_conversation(conversation, tokenizer, **kwargs)
        except ConversationFiltered as exc:
            report["filtered_conversations"] += 1
            reasons = report["filtered_by_reason"]
            reasons[exc.reason] = reasons.get(exc.reason, 0) + 1
            report["skipped"].append({"conversation_id": cid, "document_id": conversation.get("document_id"),
                "source": conversation.get("source"), "reason": exc.reason, **exc.details})
            continue
        report["accepted_conversations"] += 1
        report["accepted_assistant_turns"] += prepared.assistant_turn_count
        report["accepted_target_tokens"] += prepared.target_token_count
        report["accepted_original_template_tokens"] += prepared.original_template_token_count
        report["accepted_document_tokens"] += prepared.document_token_count
        yield prepared
    report["complete"] = True


def prepare_conversations(conversations, tokenizer, **kwargs):
    """Materializing convenience for small checks; stream large official corpora."""
    report = {}
    values = list(iter_prepared_conversations(conversations, tokenizer, report=report, **kwargs))
    skipped = report.pop("skipped")
    return PreparedConversations(values, preparation_summary=report, skipped=skipped)
