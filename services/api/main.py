"""Neural Echo API: job submission, SSE progress stream, artifact serving.

Combines the brief's api+worker split into one process (see README) — the
TRIBE model is loaded once at startup and kept warm for the process lifetime.
"""
import asyncio
import json
import logging
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from neural_echo import compat, ingest
from services.api.jobs import JOBS_DIR, JobManager, _generation_to_dict

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Configurable via env so deployments can bake this into the image at a path
# outside any mounted persistent disk — a disk mounted at the same path would
# otherwise shadow/hide whatever the Dockerfile COPY'd in (see render.yaml).
CALIBRATION_BUNDLE_PATH = Path(
    os.environ.get("CALIBRATION_BUNDLE_PATH", "data/clip_library/calibration_bundle.npz")
)

job_manager = JobManager(CALIBRATION_BUNDLE_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CALIBRATION_BUNDLE_PATH.exists():
        logger.warning(
            "No calibration bundle at %s — run scripts/build_clip_library.py before creating real jobs.",
            CALIBRATION_BUNDLE_PATH,
        )
    logger.info("Loading TRIBE model (warm singleton)...")
    compat.get_tribe_model()
    logger.info("TRIBE model ready.")
    yield


app = FastAPI(title="Neural Echo API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda_available": compat.has_cuda(),
        "calibration_bundle_present": CALIBRATION_BUNDLE_PATH.exists(),
        "license_note": "TRIBE v2 is CC-BY-NC-4.0 — this is a research demo, non-commercial use only.",
    }


@app.post("/jobs")
async def create_job(
    constraint_text: str = Form(...),
    youtube_url: str | None = Form(None),
    file: UploadFile | None = File(None),
    dry_run: bool = Form(False),
    batch_size: int = Form(10),
    max_generations: int = Form(6),
    adherence_tau: float = Form(0.15),
):
    if not youtube_url and not file:
        raise HTTPException(400, "Provide either youtube_url or file")
    if youtube_url and file:
        raise HTTPException(400, "Provide only one of youtube_url or file")

    tmp_dir = JOBS_DIR / "_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if file:
        reference_path = tmp_dir / file.filename

        def _write_upload():
            with open(reference_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

        await asyncio.to_thread(_write_upload)
    else:
        # ToS note: downloading YouTube audio violates YouTube's Terms of
        # Service. This path is a demo affordance — the file-upload path
        # above is the ToS-clean primary route (see README).
        reference_path = await asyncio.to_thread(ingest.download_youtube_audio, youtube_url, tmp_dir)

    job = job_manager.create_job(
        reference_path=reference_path,
        constraint_text=constraint_text,
        dry_run=dry_run,
        batch_size=batch_size,
        max_generations=max_generations,
        adherence_tau=adherence_tau,
    )
    return {"job_id": job.id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {**job.to_status_dict(), "result": job.result}


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    job = job_manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def event_stream():
        # job.generations (an append-only list, safe to read across threads
        # under the GIL) is the source of truth, polled by index — NOT
        # job.events (a single shared Queue.get(), which would drop or
        # duplicate events across multiple concurrent/reconnecting clients,
        # since each item can only be consumed once by whichever consumer
        # calls get() first). This also gives replay-on-reconnect for free:
        # a client connecting mid-run or after completion just starts
        # polling from index 0 and catches up immediately.
        yield f"data: {json.dumps(job.to_status_dict())}\n\n"

        sent = 0
        last_status = job.status
        while True:
            if job.status != last_status:
                last_status = job.status
                yield f"data: {json.dumps({'type': 'status', 'status': last_status})}\n\n"

            generations = job.generations
            while sent < len(generations):
                yield f"data: {json.dumps(_generation_to_dict(generations[sent]))}\n\n"
                sent += 1

            if job.status == "done":
                yield f"data: {json.dumps({'type': 'done', 'result': job.result})}\n\n"
                return
            if job.status == "error":
                yield f"data: {json.dumps({'type': 'error', 'error': job.error})}\n\n"
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/jobs/{job_id}/artifacts/{filename}")
def get_artifact(job_id: str, filename: str):
    job = job_manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # candidate audio lives in data/generated/, named by content hash
    candidate_path = Path("data/generated") / filename
    if candidate_path.exists():
        return FileResponse(candidate_path, media_type="audio/mpeg")
    job_local_path = job.job_dir / filename
    if job_local_path.exists():
        return FileResponse(job_local_path, media_type="audio/wav")
    raise HTTPException(404, "Artifact not found")
