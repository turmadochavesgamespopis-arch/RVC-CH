import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)


def save_job(job_data: Dict) -> None:
    job_id = job_data["job_id"]
    with open(JOBS_DIR / f"{job_id}.json", "w") as f:
        json.dump(job_data, f, indent=4)


def load_job(job_id: str) -> Optional[Dict]:
    path = JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def update_job(job_id: str, **kwargs) -> None:
    job = load_job(job_id)
    if job:
        job.update(kwargs)
        save_job(job)


def list_jobs() -> List[Dict]:
    jobs = []
    for path in JOBS_DIR.glob("*.json"):
        with open(path, "r") as f:
            jobs.append(json.load(f))
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return jobs


def delete_job(job_id: str) -> None:
    (JOBS_DIR / f"{job_id}.json").unlink(missing_ok=True)
