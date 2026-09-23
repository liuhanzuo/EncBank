"""Use the installed gpu-node1 Codex with the user's authorized service credential.

The key travels only on encrypted SSH stdin and in the child environment; it is
not written to disk, included in arguments, or installed as persistent login.
"""
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from codex_judge_client import BASE, MODEL, ROOT
from probe_judge_service import conversation_key

REMOTE_PYTHON = '/srv/encbank/Paper_Evolve/.venv/bin/python'
REMOTE_CODE = r'''
import json, os, pathlib, re, subprocess, sys, tempfile, time
payload = json.load(sys.stdin)
key = payload.pop('key')
assert payload['model'] == 'gpt-6-astra'
assert payload['base_url'] == 'https://sbtunnel.xiaoaojianghu.fun/v1'
env = os.environ.copy()
env['MIDCACHE_JUDGE_API_KEY'] = key
exe = str(pathlib.Path.home() / '.local/bin/codex')
version = subprocess.run([exe, '--version'], capture_output=True, text=True,
                         check=True, timeout=10).stdout.strip()
def scrub(value):
    return re.sub(r'sk-[A-Za-z0-9_-]{24,}', '[REDACTED]', value.replace(key, '[REDACTED]'))
started = time.monotonic()
with tempfile.TemporaryDirectory(prefix='midcache-judge-') as task:
    task = pathlib.Path(task)
    work = task / 'empty-workspace'
    work.mkdir()
    answer_file = task / 'answer.txt'
    args = [exe, 'exec', '--ignore-user-config', '--skip-git-repo-check',
            '--ephemeral', '--sandbox', 'read-only', '--json', '--color', 'never',
            '--model', payload['model'], '--cd', str(work),
            '--output-last-message', str(answer_file)]
    config = {
        'model_provider': 'midcache_sbtunnel',
        'model_providers.midcache_sbtunnel.name': 'MidCache authorized judge service',
        'model_providers.midcache_sbtunnel.base_url': payload['base_url'],
        'model_providers.midcache_sbtunnel.env_key': 'MIDCACHE_JUDGE_API_KEY',
        'model_providers.midcache_sbtunnel.wire_api': 'responses',
        'model_providers.midcache_sbtunnel.request_max_retries': 0,
        'model_providers.midcache_sbtunnel.stream_max_retries': 0,
        'model_providers.midcache_sbtunnel.stream_idle_timeout_ms': 60000,
        'model_reasoning_effort': 'low', 'model_reasoning_summary': 'none',
        'approval_policy': 'never', 'project_doc_max_bytes': 0,
        'project_root_markers': [], 'web_search': 'disabled',
    }
    for name, value in config.items():
        args += ['-c', name + '=' + json.dumps(value)]
    for feature in ('shell_tool', 'unified_exec', 'multi_agent', 'apps', 'plugins',
                    'browser_use', 'browser_use_external', 'computer_use',
                    'image_generation', 'view_image', 'code_mode_host', 'sleep_tool',
                    'workspace_dependencies', 'skill_search', 'memories'):
        args += ['--disable', feature]
    args += ['-']
    try:
        run = subprocess.run(args, input=payload['prompt'], capture_output=True,
                             text=True, encoding='utf-8', env=env, cwd=work,
                             timeout=payload['timeout'])
        stdout, stderr, code = run.stdout, run.stderr, run.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, code = exc.stdout or '', exc.stderr or '', None
        if isinstance(stdout, bytes): stdout = stdout.decode('utf-8', 'replace')
        if isinstance(stderr, bytes): stderr = stderr.decode('utf-8', 'replace')
    answer = answer_file.read_text(encoding='utf-8') if answer_file.exists() else ''
    print(json.dumps(dict(stdout=scrub(stdout), stderr=scrub(stderr),
                          answer=scrub(answer), returncode=code, client=version,
                          elapsed_s=time.monotonic()-started, remote_user=os.environ.get('USER'))))
'''


def call(prompt, output_dir, timeout=120):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = conversation_key()
    payload = dict(prompt=prompt, key=key, model=MODEL, base_url=BASE, timeout=timeout)
    command = REMOTE_PYTHON + ' -c ' + shlex.quote(REMOTE_CODE)
    start = time.monotonic()
    try:
        response = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'gpu-node1', command],
            input=json.dumps(payload), text=True, encoding='utf-8', capture_output=True,
            timeout=timeout+30, creationflags=subprocess.CREATE_NO_WINDOW)
        if response.returncode:
            raise RuntimeError('SSH runner failed: ' + response.stderr[:500])
        remote = json.loads(response.stdout)
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        remote = dict(stdout='', stderr=str(exc), answer='', returncode=None,
                      client='remote Codex; no completed response', remote_user=None)
    def scrub(value):
        return re.sub(r'sk-[A-Za-z0-9_-]{24,}', '[REDACTED]', value.replace(key, '[REDACTED]'))
    stdout, stderr, answer = (scrub(remote[k]) for k in ('stdout', 'stderr', 'answer'))
    (out / 'events.jsonl').write_text(stdout, encoding='utf-8')
    (out / 'stderr.txt').write_text(stderr, encoding='utf-8')
    (out / 'answer.txt').write_text(answer, encoding='utf-8')
    events = []
    for line in stdout.splitlines():
        try: events.append(json.loads(line))
        except ValueError: pass
    tools = [e for e in events if e.get('item', {}).get('type') in
             ('command_execution', 'file_change', 'mcp_tool_call', 'web_search')]
    result = dict(model_requested=MODEL, base_url=BASE, client=remote['client'],
                  host='gpu-node1', remote_user=remote['remote_user'],
                  returncode=remote['returncode'], elapsed_s=time.monotonic()-start,
                  answer=answer, tool_actions=len(tools),
                  turn_completed=any(e.get('type') == 'turn.completed' for e in events),
                  usage=[e.get('usage') for e in events if e.get('type') == 'turn.completed'],
                  errors=[e for e in events if e.get('type') in ('error', 'turn.failed')],
                  credential_persisted=False, global_configuration_changed=False)
    result['ok'] = result['returncode'] == 0 and bool(answer.strip()) and result['turn_completed'] and not tools
    if not events and stderr:
        result['errors'].append(dict(type='runner_error', message=stderr[:1000]))
    (out / 'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result
