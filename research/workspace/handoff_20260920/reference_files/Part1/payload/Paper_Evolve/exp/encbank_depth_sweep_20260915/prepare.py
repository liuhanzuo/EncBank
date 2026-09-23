"""Prepare a fixed depth grid using the completed pilot's exact inputs."""
import collections
import datetime
import gzip
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
PILOT = Path('/srv/encbank/encbank_new_backbones_20260915')
MODELS = {
    'Qwen3.5-9B': {'L': 32, 'depths': [5, 8, 11, 12, 16], 'anchor_j': 11, 'anchor_job': 33917},
    'Qwen3.8-27B': {'L': 64, 'depths': [11, 16, 20, 21, 24, 32], 'anchor_j': 21, 'anchor_job': 33916},
}


def main():
    assert not (ROOT / 'plan.json').exists(), 'Do not silently replace an existing sweep plan'
    support = ROOT / 'pilot_support'
    support.mkdir(exist_ok=True)
    for name in ('run_pilot.py', 'hybrid_reader.py', 'verify_results.py'):
        shutil.copy2(PILOT / name, support / name)
    plans = {}
    for name, item in MODELS.items():
        anchor = PILOT / 'results' / name / f"job_{item['anchor_job']}"
        protocol = json.loads((anchor / 'protocol.json').read_text())
        verified = json.loads((anchor / 'verified_summary.json').read_text())
        assert protocol['j'] == item['anchor_j'] and protocol['training']['steps'] == 200
        assert protocol['seed'] == 20260915 and protocol['n_per_synthetic_cell'] == 10
        assert verified['verified'] and verified['samples'] == 120 and verified['predictions'] == 480
        with gzip.open(anchor / 'samples.jsonl.gz', 'rt', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
        assert len(samples) == len({s['id'] for s in samples}) == 120
        counts = collections.Counter((s['task'], s['length']) for s in samples)
        assert len(counts) == 10 and all(v == (30 if k[0] == 'qasper' else 10) for k, v in counts.items())
        config = json.loads((PILOT / 'models' / name / 'config.json').read_text())['text_config']
        assert config['num_hidden_layers'] == item['L']
        types = config['layer_types']
        assert types == ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'] * (item['L'] // 4)
        state = torch.load(anchor / 'adapter-final.pt', map_location='cpu', weights_only=True)
        assert state['j'] == item['anchor_j'] and state['rank'] == 32 and state['alpha'] == 32
        nparams = sum(t.numel() for pair in state['modules'].values() for t in pair.values())
        del state
        plans[name] = {**item, 'model': str(PILOT / 'models' / name), 'anchor': str(anchor),
                       'anchor_trainable_parameters': nparams, 'layer_types': types,
                       'samples': 120, 'predictions_per_depth': 480}
    # Interleave sizes. Each point gets its own one-GPU job and fresh adapter.
    pending = {name: [j for j in item['depths'] if j != item['anchor_j']] for name, item in MODELS.items()}
    jobs = []
    for i in range(max(map(len, pending.values()))):
        for name, depths in pending.items():
            if i < len(depths):
                jobs.append({'index': len(jobs), 'name': name, 'j': depths[i]})
    plan = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'models': plans, 'jobs': jobs, 'steps': 200, 'window': 2048, 'seed': 20260915,
            'training_data': str(PILOT / 'data' / 'pg19_train_64.jsonl'),
            'pilot_only': True, 'infrastructure_measurement': False,
            'grid_rationale': 'Rounded L/6, L/4, 0.33L, L/2, plus nearby complete four-block hybrid cycles; fixed before sweep outcomes.',
            'base_replay_reuse': 'Same saved tokenized evidence and model; copy unchanged base outputs, with three exact generation spot checks at every new depth.',
            'limitations': 'One seed, 200 steps, n=10 per synthetic task/length and n=30 Qasper. Suffix LoRA spans and parameter counts vary with j. No held-out optimum claim.'}
    (ROOT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'prepared': True, 'new_jobs': len(jobs), 'depths': {k: v['depths'] for k, v in plans.items()}}))


if __name__ == '__main__':
    main()
