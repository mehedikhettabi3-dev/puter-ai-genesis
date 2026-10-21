"""Thread-safe job store used by the inference server (no GPU imports)."""
from __future__ import annotations

import threading
import time
import uuid

TERMINAL = {"completed", "failed", "cancelled"}


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def create(self, kind: str, params: dict) -> dict:
        now = time.time()
        job = {
            "id": uuid.uuid4().hex[:12],
            "kind": kind,
            "status": "queued",
            "progress": 0.0,
            "message": "",
            "params": params,
            "result": None,
            "error": None,
            "created": now,
            "updated": now,
        }
        with self._lock:
            self._jobs[job["id"]] = job
            return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j["created"], reverse=True)[:limit]
            return [dict(j) for j in jobs]

    def update(self, job_id: str, **fields) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.update(fields)
            job["updated"] = time.time()
            return dict(job)

    def cancel(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job["status"] not in TERMINAL:
                job["status"] = "cancelled"
                job["updated"] = time.time()
            return dict(job)
