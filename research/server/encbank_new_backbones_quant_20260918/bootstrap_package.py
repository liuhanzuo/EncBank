"""Freeze proven sources; adapt only rank/steps and checkpoint retention."""
from pathlib import Path
import ast, hashlib, json, shutil

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'encbank_new_backbones_formal_20260915'
CODEC = Path('/srv/encbank/legacy_workspace/paper_autonomous_multifork_iteration/evidence/encbank_honly_formal_20260911/runtime/extracted_minmax.py')

def put(name, text):
    (ROOT/name).write_text(text, encoding='utf-8', newline='\n')

def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)

def main():
    bindings = {}
    for p in [OLD/'common.py', OLD/'hybrid_reader.py', OLD/'train_formal.py', OLD/'evaluate.py', CODEC]:
        bindings[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    common = (OLD/'common.py').read_text(encoding='utf-8')
    common = replace(common, 'STEPS, WINDOW, CHUNK, SHARDS = 4000, 4096, 512, 4', 'STEPS, WINDOW, CHUNK, SHARDS = 8000, 4096, 512, 4')
    put('common.py', common)
    reader = (OLD/'hybrid_reader.py').read_text(encoding='utf-8')
    reader = replace(reader, 'def attach(self):', 'def attach(self, rank=32, alpha=32):')
    reader = replace(reader, 'wrapped = LoRALinear(module)', 'wrapped = LoRALinear(module, rank=rank, alpha=alpha)')
    # Avoid the unused legacy save() method's hard-coded rank metadata.
    reader = reader[:reader.index('    def save(self, path):')]
    put('hybrid_reader.py', reader)
    train = (OLD/'train_formal.py').read_text(encoding='utf-8')
    train = replace(train, '4,000-step PG-19 suffix distillation', '8,000-step rank-128 PG-19 suffix distillation')
    train = replace(train, 'params = reader.attach()', 'params = reader.attach(rank=128, alpha=128)')
    train = replace(train, 'rank=32, alpha=32', 'rank=128, alpha=128', count=3)
    train = replace(train, "checkpoint_selection='fixed final step 4000'", "checkpoint_selection='fixed final step 8000; step4000 diagnostic only'")
    train = replace(train, "save(out/f'adapter-step{completed}.pt', adapter)", "if completed in (4000, 8000):\n                    save(out/f'adapter-step{completed}.pt', adapter)")
    # The optimization/loss/order remain the original implementation. Supply verified
    # original corpus bytes rather than re-tokenize training data unnecessarily.
    train = replace(train, "seed_all(); tok = tokenizer(cfg)", "from binding import verify_model, seed_training_tokens\n        verify_model(cfg)\n        seed_training_tokens(cfg, out)\n        seed_all(); tok = tokenizer(cfg)")
    train = replace(train, "dump(out/'complete.json', dict(complete=True, steps=completed,", "from binding import sha256\n            dump(out/'complete.json', dict(adapter_sha256=sha256(out/'adapter-final.pt'), complete=True, steps=completed,")
    put('train_large.py', train)
    source = (OLD/'evaluate.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'scores')
    put('scoring.py', 'from eval import ruler, longeval, longbench, locomo\nimport prepare_babilong as babi\n\n' + ast.get_source_segment(source, fn) + '\n')
    shutil.copyfile(CODEC, ROOT/'extracted_minmax.py')
    put('source_bindings.json', json.dumps(bindings, indent=2)+'\n')
    # Confirm that the mathematical training loss was not edited.
    original_loss = next(n for n in ast.parse((OLD/'train_formal.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'loss_for')
    new_loss = next(n for n in ast.parse(train).body if isinstance(n, ast.FunctionDef) and n.name == 'loss_for')
    assert ast.dump(original_loss) == ast.dump(new_loss)
    print('Frozen codec, reader, unchanged distillation loss and scorers prepared.')

if __name__ == '__main__': main()
