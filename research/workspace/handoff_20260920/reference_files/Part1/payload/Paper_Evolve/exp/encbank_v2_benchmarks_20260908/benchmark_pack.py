"""Explicit complete-query token packs for the bare official benchmark templates."""
import torch
from encbank import selectors

BOUNDARY = "<ENCBANK_CONTEXT_QUERY_BOUNDARY_20260908>"


def tokenize_explicit_prompt(tok, marked_prompt, question, chunk_size, selector, topk,
                            dense_retriever=None, iter_rounds=0, iter_hop_topk=2,
                            iter_score="meanpool"):
    if marked_prompt.count(BOUNDARY) != 1:
        raise ValueError("Ambiguous or missing context/query boundary")
    context, query = marked_prompt.split(BOUNDARY)
    context_ids = tok.encode(context, add_special_tokens=True)
    query_ids = tok.encode(query, add_special_tokens=False)
    if not query_ids:
        raise ValueError("The official template has no complete trailing query")
    ids = torch.tensor([context_ids + query_ids], dtype=torch.long)
    chunks = list(ids[0, :len(context_ids)].split(chunk_size)) if context_ids else []
    question_ids = tok.encode(question.strip(), add_special_tokens=False)
    indices = selectors.select_context_chunk_indices(selector, chunks, question_ids, topk,
        None, iter_rounds=iter_rounds, iter_hop_topk=iter_hop_topk, iter_score=iter_score,
        dense_retriever=dense_retriever, dense_tokenizer=tok)
    pack = {"input_tokens": int(ids.numel()), "context_tokens": len(context_ids),
        "query_tokens": len(query_ids), "context_chunks": len(chunks),
        "selected_indices": list(indices),
        "read_pack_tokens": 1 + len(query_ids) + sum(len(chunks[i]) for i in indices),
        "truncation": "none", "chat_template": False,
        "context_query_boundary": "explicit; independently tokenized; no padding"}
    return ids, len(context_ids), list(indices), pack
