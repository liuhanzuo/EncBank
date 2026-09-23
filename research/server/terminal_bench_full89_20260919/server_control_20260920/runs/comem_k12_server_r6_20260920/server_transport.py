"""Authenticated cluster-local RPC. No SSH tunnel or Windows broker."""
import base64
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from live_mailbox import client


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}-{time.time_ns()}')
    for attempt in range(5):
        try:
            with temporary.open('wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def verify_response(proof, request_id, remote_root, arm):
    expected = str(Path(remote_root) / ('run_' + arm) / 'mailbox' / (request_id + '.response.json'))
    assert proof['path'] == expected, 'Reply belongs to another mailbox'
    data = base64.b64decode(proof['payload_base64'], validate=True)
    assert len(data) == proof['bytes']
    assert hashlib.sha256(data).hexdigest() == proof['sha256']
    response = json.loads(data)
    assert response['request_id'] == request_id, 'Reply belongs to another request'
    return data


class Transport:
    def __init__(self, home, plan, output):
        self.home, self.plan, self.output = Path(home), plan, Path(output)
        self.run = self.home / ('run_' + plan['arm'])
        self.cursor = 0

    def ctl(self, action, rid=None):
        status = {name: json.loads((self.run / name).read_text()) for name in
                  ['worker_ready.json', 'worker_failure.json', 'process_receipt.json',
                   'memory_cap_failure.json', 'worker_complete.json'] if (self.run / name).exists()}
        endpoint = self.run / 'transport_endpoint.json'
        if action == 'status' and (not endpoint.exists() or any(name in status for name in
                                    ['worker_failure.json', 'process_receipt.json'])):
            return status
        query = {'action': action}
        if rid is not None:
            query['id'] = rid
        if action == 'status':
            query['event_cursor'] = self.cursor
        result = client(json.loads(endpoint.read_text()), query, timeout=30)
        if 'transport' in result:
            transport = result['transport']
            save(self.output / ('transport_status_' + str(time.time_ns()) + '.json'), transport)
            save(self.output / 'transport_snapshot.json', transport)
            self.cursor = transport['event_cursor']
        return result

    def handle(self, request_path, child):
        request_path = Path(request_path)
        request = json.loads(request_path.read_text())
        rid = request['request_id']
        box = request_path.parent
        endpoint = json.loads((self.run / 'transport_endpoint.json').read_text())
        started = time.monotonic()
        cancelled = False
        try:
            # Publish exactly once. An uncertain acknowledgement must never trigger another generation.
            client(endpoint, {'action': 'publish', 'id': rid, 'request': dict(request)}, timeout=60)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(client, endpoint, {'action': 'wait', 'id': rid},
                                     request['remaining_seconds'] + 180)
                while not future.done():
                    if not cancelled and ((box / (rid + '.cancel.json')).exists() or child.poll() is not None):
                        self.ctl('cancel', rid)
                        cancelled = True
                    time.sleep(.1)
                proof = future.result()
            data = verify_response(proof, rid, self.plan['remote_root'], self.plan['arm'])
            response = json.loads(data)
            assert response['task'] == request['task'] and response['step'] == request['step']
            target = box / (rid + '.response.json')
            temporary = target.with_suffix('.download')
            # Retry storage only, retaining the same verified bytes.
            for attempt in range(5):
                try:
                    with temporary.open('wb') as stream:
                        stream.write(data); stream.flush(); os.fsync(stream.fileno())
                    temporary.replace(target)
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            proof = {k: v for k, v in proof.items() if k != 'payload_base64'}
            save(box / (rid + '.broker.json'), {'status': 'delivered', 'proof': proof,
                 'seconds': time.monotonic() - started, 'cancel_sent': cancelled,
                 'transport': 'authenticated-cluster-http', 'owner_host': os.uname().nodename})
        except BaseException as exc:
            try:
                self.ctl('cancel', rid)
            except Exception:
                pass
            save(box / (rid + '.error.json'), {'error': repr(exc), 'epoch': time.time(),
                 'no_automatic_request_retry': True, 'classification': 'INFRASTRUCTURE_FAILURE'})
            raise
