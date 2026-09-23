import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from select_official_pilot import choose, quotas, main


def rows():
    result = []
    for split in ("train", "dev"):
        for source in ("book", "paper", "summary"):
            for i in range(12):
                identity = f"{split}-{source}-{i}"
                result.append({"id": identity, "source": source, "split": split,
                    "document_id": identity, "overlap_group_id": f"{split}-{source}-{i//2}",
                    "tokenizer_fingerprint": "test", "original_template_token_count": 15000+i,
                    "raw_uncompressed_read_pack_tokens": 15001+i, "document_token_count": 14000+i,
                    "reader_template_token_count": 1000+i, "target_token_count": 500+i,
                    "assistant_turn_count": 2})
    return result


class PilotChecks(unittest.TestCase):
    def test_proportional_quotas_respect_budget_and_source_capacity(self):
        self.assertEqual(quotas({"a": 1, "b": 100, "c": 100}, 101), {"a": 1, "b": 50, "c": 50})
        with self.assertRaises(ValueError):
            quotas({"a": 1}, 2)

    def test_input_order_independent_and_distinct_development_groups(self):
        a = choose(rows(), train_count=24, dev_per_source=3)
        b = choose(reversed(rows()), train_count=24, dev_per_source=3)
        self.assertEqual(a, b)
        self.assertEqual(len(a["train"]), 24)
        self.assertEqual(len(a["dev"]), 9)
        self.assertEqual(len({r["overlap_group_id"] for r in a["dev"]}), 9)
        self.assertEqual({r["source"] for r in a["probe_train"]}, {"book", "paper", "summary"})
        self.assertTrue(all(r["id"].endswith("-11") for r in a["probe_train"]))

    def test_cross_split_groups_and_mixed_tokenizer_are_rejected(self):
        data = rows()
        data[-1]["overlap_group_id"] = data[0]["overlap_group_id"]
        with self.assertRaisesRegex(ValueError, "overlapping overlap_group_id"):
            choose(data, train_count=20)
        data = rows()
        data[-1]["tokenizer_fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "Mixed tokenizer"):
            choose(data, train_count=20)

    def test_file_selection_preserves_extreme_probe_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = rows()
            for row in data:
                row['conversation_id'] = row['id']
            pool, index, out = root/'pool.jsonl', root/'index.jsonl', root/'pilot'
            serialized = ''.join(json.dumps(row)+'\n' for row in data)
            pool.write_text(serialized, encoding='utf-8')
            index.write_text(serialized, encoding='utf-8')
            arguments = ['select', '--pool', str(pool), '--index', str(index),
                         '--out', str(out), '--train-count', '24', '--dev-per-source', '3']
            with patch('sys.argv', arguments), contextlib.redirect_stdout(io.StringIO()):
                main()
            probe = [json.loads(line) for line in (out/'probe_train.jsonl').read_text().splitlines()]
            selected = json.loads((out/'probe_selection.json').read_text())
            self.assertEqual(selected['train_ids'], [row['id'] for row in probe])
            self.assertEqual(probe[0]['raw_uncompressed_read_pack_tokens'],
                             max(row['raw_uncompressed_read_pack_tokens'] for row in data if row['split']=='train'))
            self.assertEqual(pool.read_text(encoding='utf-8'), serialized)
            bounded = root/'bounded'
            arguments = ['select', '--pool', str(pool), '--index', str(index),
                         '--out', str(bounded), '--train-count', '12', '--dev-per-source', '3',
                         '--max-raw-pack-tokens', '15005']
            with patch('sys.argv', arguments), contextlib.redirect_stdout(io.StringIO()):
                main()
            receipt = json.loads((bounded/'selection.json').read_text())
            resource = receipt['selection']['common_resource_subset']
            self.assertEqual(resource['eligible_before'], 72)
            self.assertEqual(resource['eligible_after'], 30)
            original = {row['id']: row for row in data}
            for filename in ('train', 'dev', 'probe_train', 'probe_dev'):
                for line in (bounded/f'{filename}.jsonl').read_text().splitlines():
                    selected_row = json.loads(line)
                    self.assertLessEqual(selected_row['raw_uncompressed_read_pack_tokens'], 15005)
                    self.assertEqual(selected_row, original[selected_row['id']])


if __name__ == "__main__":
    unittest.main(verbosity=2)
