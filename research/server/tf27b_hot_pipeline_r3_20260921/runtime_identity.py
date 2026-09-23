"""Separate immutable training identity from a verified runtime model location."""
from pathlib import Path


def resolve_configs(plan, training, checkpoint=None):
    identity = dict(training)
    assert identity['name'] == 'Qwen3.8-27B'
    assert identity['j'] == plan['j'] == 21
    assert identity['L'] == plan['model_layers'] == 64
    assert identity['revision'] == plan['model_revision']
    assert identity == plan['training_model_identity']
    runtime = dict(identity, path=plan['model'])
    Path(runtime['path']).resolve().relative_to(Path('/srv/encbank').resolve())
    if checkpoint is not None:
        assert checkpoint['step'] == 4000
        assert checkpoint['j'] == identity['j']
        assert checkpoint['model'] == identity, 'Checkpoint training identity changed'
    return identity, runtime
