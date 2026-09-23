"""Child-only Codex authentication; never read global login or obtain credentials."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ENV_KEY = 'QENCBANK_JUDGE_API_KEY'
DROP = ('CODEX_THREAD_ID', 'CODEX_INTERNAL_ORIGINATOR_OVERRIDE', 'CODEX_MANAGED_BY_CODEX',
        'CODEX_EXEC_ORIGINATOR', 'CODEX_CLI_PATH', 'OPENAI_API_KEY', 'OPENAI_ORGANIZATION',
        'OPENAI_ORG_ID', 'OPENAI_PROJECT_ID', 'OPENAI_PROJECT')


def child_environment(root, batch_id):
    assert re.fullmatch(r'b[0-9]{4}', batch_id)
    root = Path(root).resolve()
    home = root / 'runtime_homes' / batch_id
    assert home.resolve().is_relative_to(root)
    env = os.environ.copy()
    assert env.get(ENV_KEY), 'Dedicated inherited API key is absent'
    home.mkdir(parents=True, exist_ok=False)
    for name in DROP:
        env.pop(name, None)
    env['CODEX_HOME'] = str(home)
    return env, home


def scrub(data, key):
    data = data.replace(key.encode(), b'[REDACTED]')
    data = data.replace(json.dumps(key, ensure_ascii=True)[1:-1].encode(), b'[REDACTED]')
    data = re.sub(rb'(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+', rb'\1[REDACTED]', data)
    return re.sub(rb'\bsk-[A-Za-z0-9_.*-]+', b'[REDACTED]', data)


def run_batch(argv, prompt, work, env, home, out, receipt):
    """Wait for this child; scrub transport errors before saving any captured output."""
    key = env[ENV_KEY]
    assert argv[argv.index('-o') + 1] == os.devnull
    assert 'cli_auth_credentials_store="ephemeral"' in argv
    receipt.update(credential_store='ephemeral', isolated_CODEX_HOME=str(home),
                   parent_auth_configuration_modified=False,
                   removed_inherited_environment_names=list(DROP),
                   final_source='last completed agent_message; CLI -o NUL')
    child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=work, env=env,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    receipt['pid'] = child.pid
    try:
        stdout, stderr = child.communicate(prompt)
    except BaseException:
        if child.poll() is None:
            child.terminate()
        stdout, stderr = child.communicate()
        receipt.update(actual_exit_code=child.returncode, actual_parent_wait=True,
                       process_exit_observed=True, interrupted_owned_child=True)
        (out / 'events.jsonl').write_bytes(scrub(stdout, key))
        (out / 'stderr.log').write_bytes(scrub(stderr, key))
        raise
    receipt.update(actual_exit_code=child.returncode, actual_parent_wait=True,
                   process_exit_observed=True)
    safe_stdout, safe_stderr = scrub(stdout, key), scrub(stderr, key)
    (out / 'events.jsonl').write_bytes(safe_stdout)
    (out / 'stderr.log').write_bytes(safe_stderr)
    receipt['captured_transport_output_redacted'] = (stdout != safe_stdout or stderr != safe_stderr)
    receipt['raw_captured_output_sha256'] = {
        'events': hashlib.sha256(stdout).hexdigest(), 'stderr': hashlib.sha256(stderr).hexdigest()}
    receipt['runtime_auth_json_exists'] = (home / 'auth.json').exists()
    receipt['cached_token_refresh_attempt'] = any(
        marker in (stdout + stderr).lower()
        for marker in (b'refresh_token_invalidated', b'failed to refresh', b'refreshing access token'))
    receipt['runtime_exact_key_persisted'] = any(
        key.encode() in p.read_bytes() for p in home.rglob('*') if p.is_file())
    assert not receipt['runtime_auth_json_exists'], 'Unexpected child auth file'
    assert not receipt['cached_token_refresh_attempt'], 'Unexpected account token refresh'
    assert not receipt['runtime_exact_key_persisted'], 'Child runtime credential persistence detected'
    assert child.returncode == 0, ('actual CLI exit nonzero', child.returncode)
    assert not receipt['captured_transport_output_redacted'], 'Successful output required redaction; stop without scoring'
    events = [json.loads(line) for line in safe_stdout.decode('utf-8-sig').splitlines() if line.strip()]
    messages = [event['item']['text'] for event in events
                if event.get('type') == 'item.completed'
                and event.get('item', {}).get('type') == 'agent_message']
    assert messages, 'No final agent message'
    value = json.loads(messages[-1])
    with (out / 'final.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
