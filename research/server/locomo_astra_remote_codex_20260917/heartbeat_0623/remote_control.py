"""One non-scoring control; accepts a credential only through inherited env.

No execution on import. The fixed output directory prevents repeat attempts.
Only this helper's own child is terminated if its 150-second wait expires.
"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "frozen_validator"
OUTPUT = HERE / "control"
ENV_KEY = "QENCBANK_JUDGE_API_KEY"
BASE_URL = "https://sbtunnel.xiaoaojianghu.fun/v1"
PREFIX = "model_providers.astra_judge_https."
PLAN_SHA = "9ab34359ac9ad4483f0f27b50d213681b0271b64ff6d0d833b0f2e0a62ce70d5"
LAUNCH_SHA = "3520509cd899ce58662398a49bf7127b6ac7b934f8007b38332382e14e8c498b"
PROMPT = ('This is an authorized transport control only. No evaluation cases are supplied '
          'and no quality judgment is requested. Return exactly {"status":"OK"}. Do not use tools.\n')
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["status"],
          "properties": {"status": {"type": "string", "enum": ["OK"]}}}


def now():
    return datetime.datetime.now().astimezone().isoformat()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def scrub(data, key):
    """Scrub exact bytes first, including JSON escapes, then generic credentials."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    data = data.replace(key.encode("utf-8"), b"[REDACTED]")
    escaped = json.dumps(key, ensure_ascii=True)[1:-1].encode("ascii")
    data = data.replace(escaped, b"[REDACTED]")
    data = re.sub(rb"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", rb"\1[REDACTED]", data)
    data = re.sub(rb"\bsk-[A-Za-z0-9_.*-]+", b"[REDACTED]", data)
    data = re.sub(rb'''(?i)((?:api[_-]?key|access[_-]?token|auth[_-]?token|authorization)\s*["']?\s*[:=]\s*["']?)[^\s,"'<>]+''',
                  rb"\1[REDACTED]", data)
    return data


def save(path, value, key):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("xb") as stream:
        stream.write(scrub(payload, key))


def admission():
    active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            if entry.stat().st_uid != os.getuid(): continue
            cmd=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf-8','replace')
            exe=(entry/'exe').resolve().name
            if exe=='codex' and ' exec ' in cmd and ('qencbank_astra' in cmd or 'locomo_astra_remote_codex' in cmd):
                active.append({'pid':int(entry.name),'name':exe})
        except (FileNotFoundError,PermissionError,ProcessLookupError): pass
    return {'checked_at':now(),'status':'CLEAR' if not active else 'BLOCKED_ACTIVE_JUDGE',
            'active_judge_processes':active,'scope':'owned remote Codex judge processes','command_lines_saved':False}


def main():
    if len(sys.argv) != 1:
        raise SystemExit("No arguments accepted; inherit QENCBANK_JUDGE_API_KEY")
    key = json.loads(sys.stdin.buffer.read())["key"]
    os.environ[ENV_KEY] = key
    if not key or not key.strip():
        raise SystemExit("Dedicated inherited API key is absent")
    if sha(SOURCE / "plan.json") != PLAN_SHA or sha(SOURCE / "launch.py") != LAUNCH_SHA:
        raise SystemExit("Frozen source binding changed")
    plan = read(SOURCE / "plan.json")
    codex = plan["codex"]
    import tomllib
    shared=Path('/shared/autoresearch/codex-home/config.toml')
    shared_hash=sha(shared)
    assert shared_hash=='dcf903eb78826c2b9c4b2394da98a3a88bbb5cf902dc28fbbf77e0718c151639'
    settings=tomllib.loads(shared.read_text())
    assert settings['model']=='gpt-6-astra' and settings['model_reasoning_effort']=='high'
    selected=settings['model_providers'][settings['model_provider']]
    assert selected['base_url']==BASE_URL and selected['wire_api']=='responses' and selected['requires_openai_auth'] is True
    exe=Path('/srv/encbank/.local/bin/codex').resolve()
    assert sha(exe)=='3188814c35471432d4123203e0eb38e5bddc60226e3d7ddf0e59e649ea140022'
    version=subprocess.run([str(exe),'--version'],capture_output=True,text=True,check=True).stdout.strip()
    assert version=='codex-cli 0.154.0'
    root=Path('/srv/encbank').resolve()
    assert HERE.resolve().is_relative_to(root)
    work=HERE/'empty_work';work.mkdir(exist_ok=False)
    codex.update(executable=str(exe),executable_sha256=sha(exe),version=version,working_directory=str(work))
    if not (plan["model"] == "gpt-6-astra" and plan["reasoning_effort"] == "high"
            and codex["version"] == "codex-cli 0.154.0"
            and sha(Path(codex["executable"])) == codex["executable_sha256"]):
        raise SystemExit("Frozen model, effort, or CLI identity check failed")
    config = codex["config"]
    auth = PREFIX + "requires_openai_auth=true"
    if config.count(auth) != 1 or any(x.startswith(PREFIX + field + "=")
                                     for field in ("base_url", "env_key") for x in config):
        raise SystemExit("Unexpected original provider configuration")
    config += [PREFIX + "base_url=" + json.dumps(BASE_URL), 'forced_login_method="api"']
    config += ['cli_auth_credentials_store="ephemeral"']
    spec = importlib.util.spec_from_file_location("frozen_control_argv", SOURCE / "launch.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    OUTPUT.mkdir(exist_ok=False)
    record = {"started_at": now(), "status": "STARTING_NONSCORING_CONTROL", "cases_sent": 0,
              "scientific_judgments": 0, "quality_evidence": False, "automatic_retry": False,
              "model_alias": plan["model"], "reasoning_effort": plan["reasoning_effort"],
              "CLI_version": codex["version"], "CLI_executable_sha256": codex["executable_sha256"],
              "source_plan_sha256": PLAN_SHA, "source_argv_and_validator_sha256": LAUNCH_SHA,
              "provider_delta": {"base_url": BASE_URL, "env_key": "CODEX_API_KEY", "requires_openai_auth": True}, "shared_config_sha256": shared_hash, "shared_auth_read": False, "authentication": "user-supplied API key via SSH stdin then child-only CODEX_API_KEY; forced API login",
              "actual_parent_wait": False, "process_exit_observed": False,
              "actual_exit_code": None, "deadline_seconds": 150,
              "event_validator": {"status": "NOT_RUN"},
              "auth_secret_values_logged": False, "original_source_plan_unchanged": True}
    process = None
    try:
        admit = admission()
        save(OUTPUT / "admission.json", admit, key)
        if admit["status"] != "CLEAR":
            raise RuntimeError("Scoped admission found an active judge")
        save(OUTPUT / "schema.json", SCHEMA, key)
        (OUTPUT / "prompt.txt").write_bytes(scrub(PROMPT, key))
        argv = launcher.argv_for(plan, {"schema": str(OUTPUT / "schema.json")}, OUTPUT)
        # Redirect CLI's independent final-message write to NUL. Rebuild final.json
        # only from the captured event stream after redaction, before disk writes.
        argv[argv.index("-o") + 1] = os.devnull
        argv.insert(2,"--ignore-rules")
        record.update(actual_argv=argv, prompt_sha256=sha(OUTPUT / "prompt.txt"),
                      schema_sha256=sha(OUTPUT / "schema.json"),
                      final_source="last completed agent_message from captured stdout; CLI -o NUL")
        env = os.environ.copy()
        child_home = HERE / "runtime_homes/proxy"
        child_home.mkdir(parents=True, exist_ok=False)
        env["CODEX_HOME"] = str(child_home)
        record["credential_store"] = "ephemeral"
        record["isolated_CODEX_HOME"] = str(child_home)
        record["existing_account_credentials_loaded"] = False
        for name in ("CODEX_THREAD_ID", "CODEX_INTERNAL_ORIGINATOR_OVERRIDE", "CODEX_MANAGED_BY_CODEX",
                     "CODEX_EXEC_ORIGINATOR", "CODEX_CLI_PATH", "OPENAI_API_KEY",
                     "OPENAI_ORGANIZATION", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_PROJECT", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "CODEX_SQLITE_HOME"):
            env.pop(name, None)
        env['CODEX_API_KEY']=key
        for name in ('TMPDIR','XDG_CACHE_HOME','CODEX_SQLITE_HOME'):
            path=child_home/name.lower();path.mkdir();env[name]=str(path)
        save(OUTPUT / "start.json", record, key)
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, cwd=codex["working_directory"], env=env,
                                   creationflags=0)
        record["pid"] = process.pid
        try:
            stdout, stderr = process.communicate(PROMPT.encode("utf-8"), timeout=150)
        except subprocess.TimeoutExpired:
            process.terminate()
            record["diagnostic_deadline_termination"] = True
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
        record.update(actual_exit_code=process.returncode, actual_parent_wait=True,
                      process_exit_observed=True, finished_at=now())
        (OUTPUT / "events.jsonl").write_bytes(scrub(stdout, key))
        (OUTPUT / "stderr.log").write_bytes(scrub(stderr, key))
        events = [json.loads(line) for line in scrub(stdout, key).decode("utf-8-sig").splitlines() if line.strip()]
        record["event_types"] = [event.get("type") for event in events]
        try:
            summary = launcher.validate_events(plan, OUTPUT / "events.jsonl")
            record["event_validator"] = {"status": "PASS", "summary": summary}
        except Exception as error:
            record["event_validator"] = {"status": "FAIL", "error_type": type(error).__name__, "detail": str(error)}
        messages = [event["item"].get("text", "") for event in events
                    if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "agent_message"]
        final = json.loads(messages[-1]) if messages else None
        if final is not None:
            save(OUTPUT / "final.json", final, key)
        record["exact_OK"] = final == {"status": "OK"}
        record["final_value"] = final
        passed = process.returncode == 0 and record["exact_OK"] and record["event_validator"]["status"] == "PASS"
        record["status"] = "CONTROL_ACTUAL0_EXACT_OK_VALIDATED" if passed else "CONTROL_FAILED_PRESERVED_NO_RETRY"
    except Exception as error:
        record.update(status="CONTROL_FAILED_PRESERVED_NO_RETRY", error_type=type(error).__name__, detail=str(error))
        if process is not None and process.poll() is None:
            process.kill()
            try:
                stdout, stderr = process.communicate(timeout=5)
                record.update(actual_exit_code=process.returncode, actual_parent_wait=True,
                              process_exit_observed=True, exception_own_process_termination=True)
                if not (OUTPUT / "events.jsonl").exists():
                    (OUTPUT / "events.jsonl").write_bytes(scrub(stdout, key))
                if not (OUTPUT / "stderr.log").exists():
                    (OUTPUT / "stderr.log").write_bytes(scrub(stderr, key))
            except Exception as cleanup_error:
                record["cleanup_error_type"] = type(cleanup_error).__name__
    if 'child_home' in locals():
        record['runtime_auth_json_exists']=(child_home/'auth.json').exists()
        record['runtime_exact_key_persisted']=any(key.encode() in p.read_bytes() for p in child_home.rglob('*') if p.is_file())
        if record['runtime_auth_json_exists'] or record['runtime_exact_key_persisted']:
            record['status']='AUTH_ISOLATION_CHECK_FAILED_NO_SCORE'
    record["finished_at"] = now()
    record["source_plan_still_matches"] = sha(SOURCE / "plan.json") == PLAN_SHA
    record["output_sha256"] = {path.name: sha(path) for path in OUTPUT.iterdir() if path.is_file()}
    save(OUTPUT / "receipt.json", record, key)
    print(scrub(json.dumps({"status": record["status"], "actual_exit_code": record["actual_exit_code"],
                           "receipt": str(OUTPUT / "receipt.json"), "receipt_sha256": sha(OUTPUT / "receipt.json")}), key).decode())
    return 0 if record["status"] == "CONTROL_ACTUAL0_EXACT_OK_VALIDATED" else 1


if __name__ == "__main__":
    sys.exit(main())
