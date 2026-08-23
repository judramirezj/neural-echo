"""Mandatory validation gate — must pass before trusting the optimizer loop.

Run with: .venv/bin/python -m pytest tests/test_metric_sanity.py -v -s

Exercises neural_echo.metric.compute_cost directly against real TRIBE
predictions. No calibration bundle needed — the metric is self-normalizing
per region (each region's distance is divided by the benchmark's own norm).
"""
import logging
from pathlib import Path

import numpy as np
import pytest

from neural_echo import atlases, compat, ingest, metric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NORM_CLIPS_DIR = Path("data/clip_library/normalized")


@pytest.fixture(scope="module")
def model():
    return compat.get_tribe_model()


@pytest.fixture(scope="module")
def regions():
    return atlases.build_lobule_regions()


@pytest.fixture(scope="module")
def library_clips():
    clips = sorted(NORM_CLIPS_DIR.glob("*.wav"))
    clips = [c for c in clips if not c.stem.startswith("_floor")]
    if len(clips) < 5:
        pytest.skip("Need at least 5 normalized library clips for discriminability test")
    return clips


def _preds(model, clip_path: Path) -> np.ndarray:
    df = model.get_events_dataframe(audio_path=str(clip_path))
    preds, _ = model.predict(events=df)
    return np.asarray(preds)


def _cost(model, regions, ref: Path, cand: Path) -> metric.CostResult:
    return metric.compute_cost(_preds(model, cand), _preds(model, ref), regions)


def test_identical_clip_near_zero(model, regions, library_clips):
    """Identical clip -> global_score ~ 0."""
    clip = library_clips[0]
    result = _cost(model, regions, clip, clip)
    logger.info("identical_clip: global_score=%.4f", result.global_score)
    assert result.global_score < 0.1, f"Identical clip should score near 0, got {result.global_score}"


def test_music_vs_silence_large_score(model, regions, library_clips):
    """Music vs. silence should score clearly worse than music vs. itself —
    silence carries no temporal arc to correlate with, and no spatial
    structure to match either."""
    music_clip = library_clips[0]
    silence_path = Path("data/clip_library/_baseline_silence.wav")
    if not silence_path.exists():
        ingest.generate_silence(silence_path)

    identical = _cost(model, regions, music_clip, music_clip)
    vs_silence = _cost(model, regions, music_clip, silence_path)
    logger.info("music_vs_silence: global_score=%.4f (identical=%.4f)", vs_silence.global_score, identical.global_score)
    assert vs_silence.global_score > identical.global_score + 0.3, (
        f"Music vs. silence ({vs_silence.global_score:.4f}) should score clearly worse than "
        f"music vs. itself ({identical.global_score:.4f})"
    )


def test_loudness_perturbation_stays_small(model, regions, library_clips, tmp_path):
    """Loudness-perturbed copy of a clip (+6dB) -> still small global_score.
    If this fails, loudness normalization (ingest.normalize_clip) is broken."""
    import soundfile as sf

    clip = library_clips[0]
    audio, sr = sf.read(str(clip))
    louder = np.clip(audio * (10 ** (6 / 20)), -1.0, 1.0)
    louder_path = tmp_path / "louder.wav"
    sf.write(str(louder_path), louder, sr, subtype="FLOAT")

    result = _cost(model, regions, clip, louder_path)
    logger.info("loudness_perturbed: global_score=%.4f", result.global_score)
    assert result.global_score < 0.3, (
        f"A loudness-perturbed copy should still score small, got {result.global_score:.4f} — "
        "check that inputs are being loudness-normalized before TRIBE sees them"
    )


def test_genre_discriminability(model, regions, library_clips):
    """For a reference, a genre-matched clip must score better (lower) than a
    genre-mismatched clip, across at least 5 reference tracks. This is the
    load-bearing test — if it fails, stop and iterate on the region set /
    windowing before trusting the optimizer built on top of this metric."""
    electronic_family = {"deep_techno", "drum_and_bass", "reggaeton", "lofi_hiphop"}
    acoustic_family = {"acoustic_folk", "jazz_trio", "classical_piano", "orchestral_cinematic"}

    def family(n):
        if n in electronic_family:
            return "electronic"
        if n in acoustic_family:
            return "acoustic"
        return "other"

    n_refs = min(5, len(library_clips))
    correct = 0
    total = 0
    results_log = []

    for i in range(n_refs):
        ref = library_clips[i]
        others = [c for j, c in enumerate(library_clips) if j != i]
        if len(others) < 2:
            continue
        scores = [(_cost(model, regions, ref, c).global_score, c) for c in others]
        scores.sort(key=lambda x: x[0])
        closest, farthest = scores[0], scores[-1]
        ref_fam = family(ref.stem)
        if ref_fam == "other":
            continue
        total += 1
        if family(closest[1].stem) == ref_fam:
            correct += 1
        results_log.append((ref.stem, closest[1].stem, closest[0], farthest[1].stem, farthest[0]))

    for row in results_log:
        logger.info("ref=%s closest=%s(%.4f) farthest=%s(%.4f)", *row)

    if total == 0:
        pytest.skip("No reference clips had a clear genre family for this proxy discriminability test")

    logger.info("Genre discriminability: %d/%d correct", correct, total)
    assert correct >= max(1, int(0.6 * total)), (
        f"Genre-matched clips should usually score better than mismatched ones; "
        f"got {correct}/{total} correct — metric may not be measuring musical content"
    )
