"""Retry atomic file replacement only; never retry scientific work or JSONL appends."""
import sys
import time
from pathlib import Path

TRANSIENT_ERRNOS = frozenset((5, 70, 110, 121))


def diagnostic(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except OSError:
        pass  # stderr may be on the same unavailable filesystem.


def atomic_retry(operation, path, value, *, optional_progress=False,
                 delays=(1, 3, 10), sleeper=time.sleep):
    """Same state/value, finite retries. Only progress.json can be omitted."""
    optional = optional_progress and Path(path).name == 'progress.json'
    schedule = (1,) if optional else delays
    for attempt in range(len(schedule) + 1):
        try:
            return operation(path, value)
        except OSError as exc:
            if exc.errno not in TRANSIENT_ERRNOS:
                raise
            if attempt < len(schedule):
                sleeper(schedule[attempt])
                continue
            if optional:
                diagnostic('Skipped optional progress snapshot after transient I/O error: ' + str(exc))
                return None
            raise


def install(common):
    original_dump, original_save = common.dump, common.save

    def resilient_dump(path, value):
        return atomic_retry(original_dump, path, value, optional_progress=True)

    def resilient_save(path, value):
        return atomic_retry(original_save, path, value)

    common.dump, common.save = resilient_dump, resilient_save
