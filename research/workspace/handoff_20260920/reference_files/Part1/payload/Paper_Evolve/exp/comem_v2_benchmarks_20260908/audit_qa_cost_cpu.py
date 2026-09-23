"""Read-only CPU validation/snapshot of the separate local QA timing queue."""
import json
from pathlib import Path
from qa_cost_runner import HERE, load_job, read_json, reusable_rows, save_json, utc_now, validate_result
from qa_cost_bootstrap import completed_attempt


def main():
    plan_path = HERE / "results/protocol/qa_cost/task_plan.json"
    summaries = []
    for planned in read_json(plan_path)["jobs"]:
        job, inputs, config = load_job(plan_path, planned["id"])
        folder = HERE / "results/local/qa_cost" / job["phase"] / job["task"] / job["arm"]
        valid = reusable_rows(folder / "attempts", config, inputs)
        complete = completed_attempt(folder, config, inputs)
        # reusable_rows validates every actual observation and keeps duplicate copies once.
        summaries.append({"job_id": job["id"], "complete": complete is not None,
            "completed_attempt": str(complete) if complete else None,
            "verified_repetitions": len(valid), "expected_repetitions": len(job["indices"])*config["repetitions"],
            "expected_unique_method_examples": len(job["indices"]),
            "all_smoke_references_equal": all(r["smoke_reference_equal"] for r in valid.values()) if job["phase"] == "smoke" else None,
            "historical_prediction_disagreements": sum(not r["historical_prediction_equal"] for r in valid.values()),
            "generated_token_range": [min(r["generated_tokens"] for r in valid.values()), max(r["generated_tokens"] for r in valid.values())] if valid else None,
            "gate_admissions": list({json.dumps(r["hardware"]["gpu_admission"], sort_keys=True): r["hardware"]["gpu_admission"] for r in valid.values()}.values())})
    report = {"checked_at": utc_now(), "purpose": "CPU validation of real local QA timing records; incomplete jobs are not full benchmark results",
        "smoke_verified": sum(r["verified_repetitions"] for r in summaries if "/smoke/" in r["job_id"]),
        "full_verified": sum(r["verified_repetitions"] for r in summaries if "/full/" in r["job_id"]),
        "full_expected": 2400, "jobs": summaries}
    save_json(HERE / "results/local/qa_cost/CPU_AUDIT.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
