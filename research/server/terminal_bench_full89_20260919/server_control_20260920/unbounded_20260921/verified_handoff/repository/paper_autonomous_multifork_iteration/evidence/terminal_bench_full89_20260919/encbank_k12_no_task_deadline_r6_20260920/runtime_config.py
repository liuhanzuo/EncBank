"""Keep checkpoint identity immutable while binding verified relocated runtime files."""
from pathlib import Path


def bind_runtime_config(training_cfg, plan, manifest):
    assert manifest['status'] == 'PASS' and manifest['identity_matches_prior'] is True
    assert training_cfg['path'] == manifest['original_model']
    assert training_cfg['name'] == Path(manifest['original_model']).name
    assert training_cfg['revision'] == plan['model_revision'] == manifest['revision']
    assert training_cfg['j'] == plan['j'] and 0 < training_cfg['j'] < training_cfg['L']
    assert plan['model'] == manifest['model']
    assert plan['adapter_path'] == manifest['adapter']
    assert plan['adapter_sha256'] == manifest['adapter_sha256']
    root = Path('/srv/encbank').resolve()
    for value in (plan['model'], plan['adapter_path']):
        Path(value).resolve().relative_to(root)
    return dict(training_cfg, path=plan['model'])
