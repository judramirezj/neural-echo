"""Neural Echo API: job submission, SSE progress stream, artifact serving.

Combines the brief's api+worker split into one process (see README) — the
TRIBE model is loaded once at startup and kept warm for the process lifetime.
"""
import asyncio
import gzip
import json
import logging
import os
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from neural_echo import compat, ingest
from neural_echo.brain_visualization import build_brain_response_figure
from services.api.jobs import JOBS_DIR, JobManager, _iteration_to_dict

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

job_manager = JobManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading TRIBE model (warm singleton)...")
    compat.get_tribe_model()
    logger.info("TRIBE model ready.")
    yield


app = FastAPI(title="Neural Echo API", lifespan=lifespan)
frontend_origins = [
    origin.strip()
    for origin in os.environ.get("FRONTEND_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda_available": compat.has_cuda(),
    }


@app.post("/jobs")
async def create_job(
    constraint_text: str = Form(...),
    youtube_url: str | None = Form(None),
    file: UploadFile | None = File(None),
    dry_run: bool = Form(False),
    max_iterations: int = Form(10),
):
    constraint_text = constraint_text.strip()
    if not constraint_text:
        raise HTTPException(400, "constraint_text cannot be empty")
    if len(constraint_text) > 2_000:
        raise HTTPException(400, "constraint_text must be 2000 characters or fewer")
    if not 1 <= max_iterations <= 20:
        raise HTTPException(400, "max_iterations must be between 1 and 20")
    if not youtube_url and not file:
        raise HTTPException(400, "Provide either youtube_url or file")
    if youtube_url and file:
        raise HTTPException(400, "Provide only one of youtube_url or file")

    tmp_dir = JOBS_DIR / "_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if file:
        safe_name = Path(file.filename or "reference-audio").name
        reference_path = tmp_dir / f"{uuid.uuid4().hex[:12]}-{safe_name}"

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
        max_iterations=max_iterations,
    )
    return {"job_id": job.id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        **job.to_status_dict(),
        "result": job.result,
        "iterations": [_iteration_to_dict(iteration) for iteration in job.iterations],
    }


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    job = job_manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def event_stream():
        # job.iterations (an append-only list, safe to read across threads
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

            iterations = job.iterations
            while sent < len(iterations):
                yield f"data: {json.dumps(_iteration_to_dict(iterations[sent]))}\n\n"
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


@app.get("/jobs/{job_id}/brain-response")
def get_brain_response(job_id: str, request: Request):
    """Animated Plotly figure for all scored optimizer iterations so far."""
    job = job_manager.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    scored = [
        r for r in job.iterations
        if r.brain_residual is not None
        and r.brain_reference_activity is not None
        and r.brain_candidate_activity is not None
    ]
    if not scored:
        raise HTTPException(404, "No scored brain-response iterations yet")

    figure, meta = build_brain_response_figure(
        scored[0].brain_reference_activity,
        [r.brain_candidate_activity for r in scored],
        [r.brain_residual for r in scored],
        [r.iteration_index for r in scored],
    )
    # Plotly 6 encodes large mesh arrays efficiently as typed-array payloads;
    # preserve its encoder instead of asking FastAPI to walk millions of items.
    payload = json.dumps(
        {"figure": json.loads(figure.to_json()), "meta": meta},
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {"Cache-Control": "no-store", "Vary": "Accept-Encoding"}
    if "gzip" in request.headers.get("accept-encoding", "").lower():
        payload = gzip.compress(payload, compresslevel=5)
        headers["Content-Encoding"] = "gzip"
    return Response(content=payload, media_type="application/json", headers=headers)
