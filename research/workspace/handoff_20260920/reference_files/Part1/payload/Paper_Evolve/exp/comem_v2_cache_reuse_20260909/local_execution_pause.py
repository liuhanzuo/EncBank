"""Persistent user pause for this experiment's local GPU execution only.

Marker existence is sufficient, including an unreadable or partial marker.
Nothing here deletes or clears it. CPU result inspection is intentionally free
to import this module without invoking the execution guard.
"""
from pathlib import Path

MARKER = "PAUSED_BY_USER.json"


class LocalExecutionPaused(RuntimeError):
    pass


def require_unpaused(experiment_root=None):
    root = Path(experiment_root) if experiment_root is not None else Path(__file__).resolve().parent
    path = root/MARKER
    if path.exists():
        raise LocalExecutionPaused(
            f"Local RTX5090 execution is paused by the user: {path}. "
            "Do not start or resume scheduling/models or remove this marker without explicit user instruction. "
            "Read-only CPU aggregation remains allowed.")
