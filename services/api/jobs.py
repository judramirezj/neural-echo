"""In-process job manager: runs each optimization as a background thread with
its own event loop. Combines the brief's separate api/worker services into one
process for a single Render web service — see README for the tradeoff.

Progress is exposed via `job.status` and `job.generations` (an append-only
list), which services/api/main.py's SSE endpoint polls by index — not a
Queue, since a shared Queue.get() can only be consumed once per item and
would drop or duplicate events across multiple/reconnecting SSE clients.
"""
import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from neural_echo import calibration, ingest
from neural_echo.optimizer import CandidateResult, GenerationResult, OptimizerRun

logger = logging.getLogger(__name__)

JOBS_DIR = Path("data/jobs")
STUB_CLIPS_DIR = Path("data/clip_library/raw")


def _candidate_to_dict(c: CandidateResult) -> dict:
    return {
        "genome": c.genome.model_dump(mode="json"),
        "audio_path": Path(c.audio_path).name if c.audio_path else None,
        "D_brain": c.D_brain,
        "percentile": c.percentile,
        "d_spatial": c.d_spatial,
        "d_dynamics": c.d_dynamics,
        "d_geometry": c.d_geometry,
        "adherence": c.adherence,
        "novelty_audio_sim": c.novelty_audio_sim,
        "is_near_cover": c.is_near_cover,
        "passed_constraint": c.passed_constraint,
        "rejected_reason": c.rejected_reason,
        "per_network_deltas": c.per_network_deltas,
    }


def _generation_to_dict(g: GenerationResult) -> dict:
    return {
        "type": "generation_complete",
        "generation_index": g.generation_index,
        "hypothesis": g.hypothesis,
        "learned_insights": g.learned_insights,
        "candidates": [_candidate_to_dict(c) for c in g.candidates],
        "best": _candidate_to_dict(g.best) if g.best else None,
        "mean_D_brain": g.mean_D_brain,
        "elapsed_s": g.elapsed_s,
    }


@dataclass
class Job:
    id: str
    status: str = "pending"  # pending | preparing | running | done | error
    error: str | None = None
    generations: list = field(default_factory=list)
    result: dict | None = None
    job_dir: Path = field(default_factory=lambda: JOBS_DIR)
    constraint_text: str = ""
    dry_run: bool = False
    created_at: float = field(default_factory=time.time)

    def to_status_dict(self) -> dict:
        return {
            "id": self.id, "status": self.status, "error": self.error,
            "n_generations": len(self.generations), "constraint_text": self.constraint_text,
            "dry_run": self.dry_run,
        }


class JobManager:
    def __init__(self, calibration_bundle_path: Path):
        self.jobs: dict[str, Job] = {}
        self._bundle_path = calibration_bundle_path
        self._bundle = None

    def _get_bundle(self) -> calibration.CalibrationBundle:
        if self._bundle is None:
            self._bundle = calibration.CalibrationBundle.load(self._bundle_path)
        return self._bundle

    def create_job(
        self,
        reference_path: Path,
        constraint_text: str,
        dry_run: bool = False,
        batch_size: int = 10,
        max_generations: int = 6,
        adherence_tau: float = 0.15,
    ) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, constraint_text=constraint_text, dry_run=dry_run, job_dir=job_dir)
        self.jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job, reference_path, constraint_text, dry_run, batch_size, max_generations, adherence_tau),
            daemon=True,
        )
        thread.start()
        return job

    def _run_job(self, job: Job, reference_path: Path, constraint_text: str, dry_run: bool,
                 batch_size: int, max_generations: int, adherence_tau: float):
        try:
            job.status = "preparing"

            normalized_path = job.job_dir / "reference.wav"
            ingest.normalize_clip(reference_path, normalized_path)

            bundle = self._get_bundle()

            def on_generation(g: GenerationResult):
                job.generations.append(g)

            run = OptimizerRun(
                reference_audio_path=str(normalized_path),
                constraint_text=constraint_text,
                bundle=bundle,
                db_path=job.job_dir / "run.sqlite3",
                dry_run=dry_run,
                stub_clips_dir=STUB_CLIPS_DIR if dry_run else None,
                batch_size=batch_size,
                max_generations=max_generations,
                adherence_tau=adherence_tau,
                on_generation=on_generation,
            )

            job.status = "running"

            asyncio.run(run.run())

            all_scored = [c for g in run.history for c in g.candidates if c.D_brain is not None]
            best = min(all_scored, key=lambda c: c.D_brain) if all_scored else None
            job.result = {
                "best": _candidate_to_dict(best) if best else None,
                "n_generations": len(run.history),
                "noise_floor": bundle.floor,
                "null_median": float(bundle.null_distribution.mean()),
                "reference_analysis": run.reference_analysis,
            }
            job.status = "done"
        except Exception as e:
            logger.exception("Job %s failed", job.id)
            job.status = "error"
            job.error = str(e)
