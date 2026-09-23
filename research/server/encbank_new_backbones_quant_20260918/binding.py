"""Small explicit provenance checks; all old files are read-only."""
import hashlib, json, os
from pathlib import Path
from common import ROOT, MODELS, TRAIN_DATA, dump

OLD = Path('/srv/encbank/encbank_new_backbones_formal_20260915')
REMOTE = Path('/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918')

def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def inference_sha(row):
    # Labels, answers and scoring-only fields never enter this digest or inference.
    keys = ('input_ids', 'selected', 'segments', 'question', 'budget', 'sink')
    return hashlib.sha256(json.dumps({k:row[k] for k in keys}, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def verify_model(cfg):
    frozen = json.loads((ROOT/'verified_inputs.json').read_text())['models'][cfg['name']]
    assert frozen['model'] == cfg
    for name, digest in frozen['model_files'].items():
        assert sha256(Path(cfg['path'])/name) == digest, name
    for name, info in frozen['weight_files'].items():
        stat = (Path(cfg['path'])/name).stat()
        assert stat.st_size == info['bytes'] and stat.st_mtime_ns == info['mtime_ns'], name
    for name, digest in frozen['training_files'].items():
        assert sha256(OLD/'training'/cfg['name']/name) == digest, name
    assert json.loads((OLD/'training'/cfg['name']/'complete.json').read_text())['steps'] == 4000
    assert json.loads((OLD/'samples'/cfg['name']/'complete.json').read_text())['samples'] == 7236
    return frozen

def seed_training_tokens(cfg, out):
    frozen = json.loads((ROOT/'verified_inputs.json').read_text())['models'][cfg['name']]
    for name in ('pg19_tokens.u32', 'tokens.json'):
        src, dst = OLD/'training'/cfg['name']/name, out/name
        assert sha256(src) == frozen['training_files'][name]
        if not dst.exists(): os.link(src, dst)
        assert sha256(dst) == frozen['training_files'][name]

def verify_code():
    manifest = json.loads((ROOT/'package_manifest.json').read_text())
    for name, digest in manifest.items(): assert sha256(ROOT/name) == digest, name

def old_predictions(cfg):
    out = {}
    for folder in ('results', 'results_locomo'):
        for p in sorted((OLD/folder/cfg['name']).glob('shard*/predictions.jsonl')):
            for line in p.open(encoding='utf-8'):
                r = json.loads(line)
                if r['arm'] != 'cache_lora': continue
                assert r['id'] not in out
                out[r['id']] = r
    assert len(out) == 7236
    return out
