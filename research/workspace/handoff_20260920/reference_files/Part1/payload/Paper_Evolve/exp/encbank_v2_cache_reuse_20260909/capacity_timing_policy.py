"""CPU-only acceptance policy for preserved but excluded timing attempts."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path

REGISTRY = "TIMING_REMEASUREMENT.json"
EXCLUSION = "TIMING_EXCLUDED.json"


def remeasurement(folder):
    path = Path(folder)/REGISTRY
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    replacement = value.get("replacement_attempt")
    if (value.get("protocol") != "capacity-complete-trace-remeasurement-v1"
            or not isinstance(replacement, str) or len(replacement) != 4 or not replacement.isdigit()
            or replacement in value.get("excluded_attempts", [])):
        raise ValueError(f"Invalid timing replacement registry: {path}")
    return value


def exclusion_reason(attempt):
    attempt = Path(attempt)
    if (attempt/EXCLUSION).exists():
        return "Timing excluded by preserved external-CPU contamination evidence"
    # A CPU note is conservative even before the explicit replacement is filed.
    if (attempt/"CPU_CONTAMINATION_NOTE.json").exists():
        return "Known external CPU overlap; complete fresh trace required"
    registry = remeasurement(attempt.parent.parent)
    if registry and attempt.name != registry["replacement_attempt"]:
        return "Only the explicitly registered fresh timing replacement is eligible"
    return None


def verify_replacement_config(attempt, config):
    registry = remeasurement(Path(attempt).parent.parent)
    if registry:
        digest = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
        if digest != registry["expected_config_sha256"]:
            raise ValueError("Registered replacement config differs from the complete original trace")


def unresolved_remeasurements(results_root):
    """Small marker scan only; this is a scheduling guard, not record validation."""
    pending = []
    for path in Path(results_root).glob("*/*/*/*/*/TIMING_REMEASUREMENT.json"):
        registry = remeasurement(path.parent)
        receipt = path.parent/"attempts"/registry["replacement_attempt"]/"REMEASUREMENT_VALIDATED.json"
        validated = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
        if not (registry.get("status") == "complete" and validated.get("status") == "complete"
                and validated.get("completed_queries") == registry.get("expected_queries")
                and validated.get("replacement_attempt") == registry["replacement_attempt"]):
            pending.append({"registry": str(path), "job_id": registry["job_id"],
                            "status": registry.get("status"), "replacement_attempt": registry["replacement_attempt"]})
    return pending
