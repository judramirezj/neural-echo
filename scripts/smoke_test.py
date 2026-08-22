"""End-to-end dry-run smoke test (brief §6 `make smoke`): exercises the full
optimizer loop — LLM genome proposals, scoring, elitism, stopping rules —
using pre-rendered stub clips instead of real ElevenLabs calls, so it costs
no ElevenLabs credits. The LLM (Claude) still runs for real; that's the part
being exercised, and it's cheap for 2 small generations.

Requires: data/clip_library/calibration_bundle.npz (run `make calibration`
once first) and ANTHROPIC_API_KEY in .env.
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from neural_echo import calibration, ingest
from neural_echo.optimizer import OptimizerRun

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BUNDLE_PATH = Path("data/clip_library/calibration_bundle.npz")
STUB_CLIPS_DIR = Path("data/clip_library/raw")
REFERENCE_CANDIDATES = [
    Path("data/clip_library/normalized/rock_anthem.wav"),
    Path("electronica_candidato.mp3"),
]


def main():
    load_dotenv()

    if not BUNDLE_PATH.exists():
        logger.error("No calibration bundle at %s — run `make calibration` first.", BUNDLE_PATH)
        sys.exit(1)

    reference = next((p for p in REFERENCE_CANDIDATES if p.exists()), None)
    if reference is None:
        logger.error("No reference clip found among %s", REFERENCE_CANDIDATES)
        sys.exit(1)

    normalized_ref = Path("data/jobs/_smoke/reference.wav")
    normalized_ref.parent.mkdir(parents=True, exist_ok=True)
    ingest.normalize_clip(reference, normalized_ref, window_s=20.0)  # short window -> fast smoke test

    bundle = calibration.CalibrationBundle.load(BUNDLE_PATH)

    def on_generation(g):
        logger.info(
            "[smoke] generation %d: hypothesis=%r best_D_brain=%s",
            g.generation_index, g.hypothesis[:200],
            f"{g.best.D_brain:.4f}" if g.best else None,
        )

    run = OptimizerRun(
        reference_audio_path=str(normalized_ref),
        constraint_text="use natural forest sounds",
        bundle=bundle,
        db_path=Path("data/jobs/_smoke/run.sqlite3"),
        dry_run=True,
        stub_clips_dir=STUB_CLIPS_DIR,
        batch_size=3,
        max_generations=2,
        on_generation=on_generation,
    )

    history = asyncio.run(run.run())

    all_scored = [c for g in history for c in g.candidates if c.D_brain is not None]
    if not all_scored:
        logger.error("[smoke] FAILED: no candidate was ever scored")
        sys.exit(1)

    best = min(all_scored, key=lambda c: c.D_brain)
    logger.info(
        "[smoke] PASSED: %d generations, %d scored candidates, best D_brain=%.4f (floor=%.4f)",
        len(history), len(all_scored), best.D_brain, bundle.floor,
    )


if __name__ == "__main__":
    main()
