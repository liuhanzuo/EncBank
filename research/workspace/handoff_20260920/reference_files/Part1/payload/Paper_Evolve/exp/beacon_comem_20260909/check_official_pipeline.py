"""Small CPU-only integration check of real released examples and Qwen3 chat.

Not a corpus-wide tokenizer audit, train/dev benchmark, or GPU resource probe.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from prepare_official_sft_data import iter_canonical_conversations
from official_sft import prepare_conversation, ConversationFiltered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--processed", type=Path,
                        default=root / "data/activation_beacon_original/processed")
    parser.add_argument("--tokenizer", default="/srv/encbank/legacy_workspace/models/Qwen3-8B")
    parser.add_argument("--output", type=Path,
                        default=root / "data/activation_beacon_original/pipeline_smoke.json")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    successes, excluded, attempts = {}, [], Counter()
    for row in iter_canonical_conversations(args.processed):
        source = row["source"]
        if source in successes or attempts[source] >= 20:
            continue
        attempts[source] += 1
        try:
            item = prepare_conversation(row, tokenizer)
        except ConversationFiltered as exc:
            excluded.append({"id": row["conversation_id"], "source": source,
                             "reason": exc.reason, "details": exc.details})
            continue
        successes[source] = {
            "id": item.id, "source_file": item.source_file,
            "source_index": item.source_index,
            "original_template_tokens": item.original_template_token_count,
            "document_tokens": item.document_token_count,
            "document_chunks": len(item.document_chunks),
            "reader_tokens": item.reader_template_token_count,
            "assistant_turns": item.assistant_turn_count,
            "supervised_tokens": item.target_token_count,
            "supervision_shift_valid": all(
                label == -100 or i in item.target_indices
                for i, label in enumerate(item.label_ids)),
            "truncated": item.protocol["truncated"],
        }
        if len(successes) == 5:
            break
    result = {"scope": "CPU parser-to-native-Qwen3 tokenizer integration; one eligible conversation per source, not full-corpus validation",
              "models_loaded": False, "gpu_used": False,
              "original_template_limits": [7200, 20000],
              "successes": successes, "filtered_while_selecting": excluded,
              "attempts_by_source": dict(attempts), "complete_five_sources": len(successes) == 5}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if len(successes) != 5:
        raise SystemExit("Not all five sources passed the CPU integration check")


if __name__ == "__main__":
    main()
