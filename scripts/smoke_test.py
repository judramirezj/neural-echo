"""End-to-end dry-run smoke test (brief §6 `make smoke`): exercises the full
optimizer loop — LLM plan proposals, brain-cost scoring, reformulation on
rejection, stopping rules — using pre-rendered stub clips instead of real
ElevenLabs calls, so it costs no ElevenLabs credits. The LLM (Claude) still
runs for real; that's the part being exercised, and it's cheap for 2 small
iterations.

Requires ANTHROPIC_API_KEY in .env. No calibration bundle needed — the
brain-cost metric (neural_echo/metric.py) is self-normalizing per region.
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from neural_echo import ingest
from neural_echo.optimizer import IterationResult, OptimizerRun

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STUB_CLIPS_DIR = Path("data/clip_library/raw")
REFERENCE_CANDIDATES = [
    Path("data/clip_library/normalized/rock_anthem.wav"),
    Path("electronica_candidato.mp3"),
]


def main():
    load_dotenv()

    reference = next((p for p in REFERENCE_CANDIDATES if p.exists()), None)
    if reference is None:
        logger.error("No reference clip found among %s", REFERENCE_CANDIDATES)
        sys.exit(1)

    normalized_ref = Path("data/jobs/_smoke/reference.wav")
    normalized_ref.parent.mkdir(parents=True, exist_ok=True)
    ingest.normalize_clip(reference, normalized_ref, window_s=20.0)  # short window -> fast smoke test

    def on_iteration(r: IterationResult):
        logger.info(
            "[smoke] iteration %d: reasoning=%r global_score=%s rejected=%s",
            r.iteration_index, r.reasoning[:200],
            f"{r.cost.global_score:.4f}" if r.cost else None, r.rejected_reason,
        )

    run = OptimizerRun(
        reference_audio_path=str(normalized_ref),
        constraint_text="use natural forest sounds",
        db_path=Path("data/jobs/_smoke/run.sqlite3"),
        dry_run=True,
        stub_clips_dir=STUB_CLIPS_DIR,
        max_iterations=2,
        on_iteration=on_iteration,
    )

    history = asyncio.run(run.run())

    scored = [r for r in history if r.cost is not None]
    if not scored:
        logger.error("[smoke] FAILED: no iteration was ever scored")
        sys.exit(1)

    best = min(scored, key=lambda r: r.cost.global_score)
    logger.info(
        "[smoke] PASSED: %d iterations, %d scored, best global_score=%.4f",
        len(history), len(scored), best.cost.global_score,
    )


if __name__ == "__main__":
    main()
