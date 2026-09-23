"""Official Codex calls on gpu-node1; credentials remain in process memory."""
import functools
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

MODEL = 'gpt-6-astra'
BASE = 'https://sbtunnel.xiaoaojianghu.fun/v1'
EXE = Path.home() / '.local/bin/codex'


@functools.lru_cache(maxsize=1)
def version():
    return subprocess.run([str(EXE), '--version'], capture_output=True, text=True,
                          check=True, timeout=10).stdout.strip()


def call(prompt, output_dir, timeout=120):
    key = os.environ['MIDCACHE_JUDGE_API_KEY']
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    def scrub(value):
        return re.sub(r'sk-[A-Za-z0-9_-]{24,}', '[REDACTED]', value.replace(key, '[REDACTED]'))
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='midcache-judge-') as work:
        args = [str(EXE), 'exec', '--ignore-user-config', '--skip-git-repo-check',
                '--ephemeral', '--sandbox', 'read-only', '--json', '--color', 'never',
                '--model', MODEL, '--cd', work,
                '--output-last-message', str(out / 'answer.txt')]
        config = {
            'model_provider': 'midcache_sbtunnel',
            'model_providers.midcache_sbtunnel.name': 'MidCache authorized judge service',
            'model_providers.midcache_sbtunnel.base_url': BASE,
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
            run = subprocess.run(args, input=prompt, capture_output=True, text=True,
                                 encoding='utf-8', cwd=work, timeout=timeout)
            stdout, stderr, code = run.stdout, run.stderr, run.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code = exc.stdout or '', exc.stderr or '', None
            if isinstance(stdout, bytes): stdout = stdout.decode('utf-8', 'replace')
            if isinstance(stderr, bytes): stderr = stderr.decode('utf-8', 'replace')
    stdout, stderr = scrub(stdout), scrub(stderr)
    (out / 'events.jsonl').write_text(stdout, encoding='utf-8')
    (out / 'stderr.txt').write_text(stderr, encoding='utf-8')
    answer = scrub((out / 'answer.txt').read_text(encoding='utf-8')) if (out / 'answer.txt').exists() else ''
    (out / 'answer.txt').write_text(answer, encoding='utf-8')
    events = []
    for line in stdout.splitlines():
        try: events.append(json.loads(line))
        except ValueError: pass
    tools = [e for e in events if e.get('item', {}).get('type') in
             ('command_execution', 'file_change', 'mcp_tool_call', 'web_search')]
    result = dict(model_requested=MODEL, base_url=BASE, client=version(),
                  host='gpu-node1', remote_user=os.environ.get('USER'), returncode=code,
                  elapsed_s=time.monotonic()-start, answer=answer, tool_actions=len(tools),
                  turn_completed=any(e.get('type') == 'turn.completed' for e in events),
                  usage=[e.get('usage') for e in events if e.get('type') == 'turn.completed'],
                  errors=[e for e in events if e.get('type') in ('error', 'turn.failed')],
                  credential_persisted=False, global_configuration_changed=False)
    result['ok'] = code == 0 and bool(answer.strip()) and result['turn_completed'] and not tools
    (out / 'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result
