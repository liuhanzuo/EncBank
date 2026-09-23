"""One empirical depth point, preserving the completed pilot's recipe."""
import argparse
import collections
import copy
import gzip
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'pilot_support'))
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from hybrid_reader import HybridReader
from run_pilot import dump, score, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, required=True)
    a = p.parse_args()
    plan = json.loads((ROOT / 'plan.json').read_text())
    task = plan['jobs'][a.index]
    assert task['index'] == a.index
    info = plan['models'][task['name']]
    j = task['j']
    assert j != info['anchor_j'] and j in info['depths']
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    assert os.environ.get('PYTHONHASHSEED') == '0'
    torch.set_num_threads(2)
    torch.manual_seed(plan['seed']); random.seed(plan['seed']); np.random.seed(plan['seed'])
    out = ROOT / 'results' / task['name'] / f'j{j:02}' / f"job_{os.environ['SLURM_JOB_ID']}"
    out.mkdir(parents=True, exist_ok=False)
    try:
        anchor = Path(info['anchor'])
        protocol = copy.deepcopy(json.loads((anchor / 'protocol.json').read_text()))
        config = AutoConfig.from_pretrained(info['model'], local_files_only=True)
        textconfig = getattr(config, 'text_config', config)
        assert textconfig.num_hidden_layers == info['L']
        cls = AutoModelForCausalLM if config.model_type == 'qwen3_5_text' else AutoModelForImageTextToText
        tok = AutoTokenizer.from_pretrained(info['model'], local_files_only=True)
        tok.model_max_length = 10**9
        sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        assert sink == protocol['sink']
        protocol.update(j=j, rule=plan['grid_rationale'], L=info['L'], normalized_depth=j/info['L'],
                        complete_hybrid_cycle=(j % 4 == 0), prefix_last_block=info['layer_types'][j-1],
                        suffix_first_block=info['layer_types'][j], suffix_layers=info['L']-j,
                        sweep_index=a.index, sweep_grid=info['depths'], anchor=str(anchor),
                        sample_source=str(anchor / 'samples.jsonl.gz'),
                        training_data=plan['training_data'], training_seed=plan['seed'],
                        base_replay_source=str(anchor / 'predictions.jsonl'),
                        limitations=plan['limitations'],
                        gpu=torch.cuda.get_device_name(), gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
                        host=platform.node(), job=os.environ['SLURM_JOB_ID'],
                        torch=torch.__version__, transformers=transformers.__version__)
        # Same runtime and training recipe as the reused anchor.
        previous = json.loads((anchor / 'protocol.json').read_text())
        assert protocol['training'] == previous['training']
        assert protocol['torch'] == previous['torch'] and protocol['transformers'] == previous['transformers']
        dump(out / 'protocol.json', protocol)
        shutil.copy2(anchor / 'samples.jsonl.gz', out / 'samples.jsonl.gz')
        with gzip.open(out / 'samples.jsonl.gz', 'rt', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        base = {r['id']: r for r in map(json.loads, (anchor / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()) if r['arm'] == 'replay_base'}
        assert len(rows) == 120 and set(base) == {r['id'] for r in rows}
        dump(out / 'progress.json', {'phase': 'loading', 'j': j, 'model': task['name']})
        model = cls.from_pretrained(info['model'], local_files_only=True, dtype=torch.bfloat16,
                                    attn_implementation='sdpa', device_map='cuda').eval()
        reader = HybridReader(model, j)
        dump(out / 'correctness.json', reader.validate(tok))
        train(reader, tok, sink, Path(plan['training_data']), out, plan['steps'])
        protocol['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        protocol['lora_module_count'] = len(reader.modules)
        dump(out / 'protocol.json', protocol)
        dump(out / 'correctness_adapted.json', reader.validate(tok))
        configured_eos = getattr(model.generation_config, 'eos_token_id', None)
        eos_ids = set(configured_eos or []) if isinstance(configured_eos, list) else {tok.eos_token_id}
        spot_checks = []
        for ri in (0, 60, 90):
            row = rows[ri]
            with reader.adapter(False), torch.inference_mode():
                ids = reader.generate(row['segments'], 'replay', row['budget'], eos_ids)
            check = {'id': row['id'], 'equal': ids == base[row['id']]['generated_ids'], 'generated_ids': ids}
            spot_checks.append(check)
            dump(out / 'base_reuse_checks.json', spot_checks)
            assert check['equal'], 'Base replay differs from saved anchor: stop before mixing results'
        counts = collections.defaultdict(list)
        arms = [('replay_base', 'replay', False), ('cache_without_lora', 'cache', False),
                ('replay_shared_lora', 'replay', True), ('cache_lora', 'cache', True)]
        with (out / 'predictions.jsonl').open('w', encoding='utf-8') as f:
            for ri, row in enumerate(rows):
                for arm, mode, enabled in arms:
                    if arm == 'replay_base':
                        record = dict(base[row['id']], reused_from=str(anchor / 'predictions.jsonl'))
                    else:
                        with reader.adapter(enabled), torch.inference_mode():
                            ids = reader.generate(row['segments'], mode, row['budget'], eos_ids)
                        text = tok.decode(ids, skip_special_tokens=True)
                        record = {'id': row['id'], 'arm': arm, 'generated_ids': ids, 'text': text,
                                  'score': score(row, text), 'task': row['task'], 'length': row['length'], 'answers': row['answers']}
                    counts[f"{row['task']}:{row['length']}:{arm}"].append(record['score'])
                    f.write(json.dumps(record, ensure_ascii=False) + '\n'); f.flush()
                dump(out / 'progress.json', {'phase': 'evaluation', 'j': j, 'completed_samples': ri+1, 'total_samples': len(rows)})
                print(json.dumps({'sample': ri+1, 'total': len(rows), 'id': row['id'], 'j': j}), flush=True)
        cells = {k: {'n': len(v), 'mean': 100*sum(v)/len(v)} for k, v in counts.items()}
        assert len(cells) == 40 and all(v['n'] == (30 if k.startswith('qasper:') else 10) for k, v in cells.items())
        dump(out / 'summary.json', {'complete': True, 'pilot_only': True, 'cells': cells})
        dump(out / 'progress.json', {'phase': 'evaluation_complete_pending_verification', 'j': j, 'completed_samples': len(rows)})
    except BaseException:
        dump(out / 'failure.json', {'traceback': traceback.format_exc(), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
