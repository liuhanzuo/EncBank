from pathlib import Path
import json, os

def dump(path, value):
    import errno, sys, time
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    tmp = path.with_name(path.name + '.tmp')
    for attempt, delay in enumerate((0, 1, 2, 4, 8)):
        if delay: time.sleep(delay)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_text(encoding='utf8') == payload: return
                raise FileExistsError('Refuse to replace a different completed artifact: ' + str(path))
            with tmp.open('w', encoding='utf8') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            tmp.replace(path)
            return
        except OSError as exc:
            if exc.errno not in (errno.EIO, getattr(errno, 'EREMOTEIO', 121)) or attempt == 4: raise
            print(json.dumps(dict(event='same_payload_persistence_retry', path=str(path),
                                  errno=exc.errno, attempt=attempt + 1)), file=sys.stderr, flush=True)
