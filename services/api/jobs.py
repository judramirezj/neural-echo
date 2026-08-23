"""In-process job manager: runs each optimization as a background thread with
its own event loop. Combines the brief's separate api/worker services into one
process for a single Render web service — see README for the tradeoff.

Progress is exposed via `job.status` and `job.iterations` (an append-only
list), which services/api/main.py's SSE endpoint polls by index — not a
Queue, since a shared Queue.get() can only be consumed once per item and
would drop or duplicate events across multiple/reconnecting SSE clients.
"""
import asyncio
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from neural_echo import ingest
from neural_echo.optimizer import IterationResult, OptimizerRun

logger = logging.getLogger(__name__)

JOBS_DIR = Path("data/jobs")
# See main.py's env-configurable path comments — same reasoning, configurable
# so it can be baked into the image outside any mounted persistent disk.
STUB_CLIPS_DIR = Path(os.environ.get("STUB_CLIPS_DIR", "data/clip_library/raw"))


def _iteration_to_dict(r: IterationResult) -> dict:
    return {
        "type": "iteration_complete",
        "iteration_index": r.iteration_index,
        "reasoning": r.reasoning,
        "changes_summary": r.changes_summary,
        "plan": r.plan.model_dump(mode="json"),
        "seed": r.seed,
        "audio_path": Path(r.audio_path).name if r.audio_path else None,
        "is_best": r.is_best,
        "elapsed_s": r.elapsed_s,
        "cost": {
            "global_score": r.cost.global_score,
            "regions": [vars(rs) for rs in r.cost.regions],
            "windows": [vars(w) for w in r.cost.windows],
            "worst_cell": vars(r.cost.worst_cell),
            "laterality": r.cost.laterality,
        } if r.cost else None,
        "rejected_reason": r.rejected_reason,
        "adherence": r.adherence,
        "novelty_audio_sim": r.novelty_audio_sim,
        "is_near_cover": r.is_near_cover,
    }


@dataclass
class Job:
    id: str
    status: str = "pending"  # pending | preparing | running | done | error
    error: str | None = None
    iterations: list = field(default_factory=list)
    result: dict | None = None
    job_dir: Path = field(default_factory=lambda: JOBS_DIR)
    constraint_text: str = ""
    dry_run: bool = False
    created_at: float = field(default_factory=time.time)

    def to_status_dict(self) -> dict:
        return {
            "id": self.id, "status": self.status, "error": self.error,
            "n_iterations": len(self.iterations), "constraint_text": self.constraint_text,
            "dry_run": self.dry_run,
        }


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}

    def create_job(
        self,
        reference_path: Path,
        constraint_text: str,
        dry_run: bool = False,
        max_iterations: int = 10,
        adherence_tau: float = 0.15,
    ) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, constraint_text=constraint_text, dry_run=dry_run, job_dir=job_dir)
        self.jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job, reference_path, constraint_text, dry_run, max_iterations, adherence_tau),
            daemon=True,
        )
        thread.start()
        return job

    def _run_job(self, job: Job, reference_path: Path, constraint_text: str, dry_run: bool,
                 max_iterations: int, adherence_tau: float):
        try:
            job.status = "preparing"

            normalized_path = job.job_dir / "reference.wav"
            ingest.normalize_clip(reference_path, normalized_path)

            def on_iteration(r: IterationResult):
                job.iterations.append(r)

            run = OptimizerRun(
                reference_audio_path=str(normalized_path),
                constraint_text=constraint_text,
                db_path=job.job_dir / "run.sqlite3",
                dry_run=dry_run,
                stub_clips_dir=STUB_CLIPS_DIR if dry_run else None,
                max_iterations=max_iterations,
                adherence_tau=adherence_tau,
                on_iteration=on_iteration,
            )

            job.status = "running"

            asyncio.run(run.run())

            scored = [r for r in run.history if r.cost is not None]
            best = min(scored, key=lambda r: r.cost.global_score) if scored else None
            job.result = {
                "best": _iteration_to_dict(best) if best else None,
                "n_iterations": len(run.history),
                "reference_analysis": run.reference_analysis,
            }
            job.status = "done"
        except Exception as e:
            logger.exception("Job %s failed", job.id)
            job.status = "error"
            job.error = str(e)
