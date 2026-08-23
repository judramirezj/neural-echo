"""Mandatory validation gate (brief §3) — must pass before Phase 3 (optimizer).

Run with: .venv/bin/python -m pytest tests/test_metric_sanity.py -v -s
Requires data/clip_library/calibration_bundle.npz (run scripts/build_clip_library.py first).
"""
import logging
from pathlib import Path

import numpy as np
import pytest

from neural_echo import calibration, compat, ingest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUNDLE_PATH = Path("data/clip_library/calibration_bundle.npz")
RAW_CLIPS_DIR = Path("data/clip_library/raw")
NORM_CLIPS_DIR = Path("data/clip_library/normalized")

pytestmark = pytest.mark.skipif(
    not BUNDLE_PATH.exists(), reason="calibration bundle not built — run scripts/build_clip_library.py"
)


@pytest.fixture(scope="module")
def bundle():
    return calibration.CalibrationBundle.load(BUNDLE_PATH)


@pytest.fixture(scope="module")
def model():
    return compat.get_tribe_model()


@pytest.fixture(scope="module")
def library_clips():
    clips = sorted(NORM_CLIPS_DIR.glob("*.wav"))
    clips = [c for c in clips if not c.stem.startswith("_floor")]
    if len(clips) < 5:
        pytest.skip("Need at least 5 normalized library clips for discriminability test")
    return clips


def _score(model, bundle, ref: Path, cand: Path):
    result, _, _ = calibration.score_against_reference(model, bundle, ref, cand)
    return result


def test_identical_clip_near_zero(model, bundle, library_clips):
    """Identical clip -> D_brain ~ 0."""
    clip = library_clips[0]
    result = _score(model, bundle, clip, clip)
    logger.info("identical_clip: D_brain=%.4f", result.D_brain)
    assert result.D_brain < 0.05, f"Identical clip should score near 0, got {result.D_brain}"


def test_same_track_excerpts_high_percentile(bundle):
    """Two non-overlapping excerpts of the same track -> D_brain small enough
    to be "closer than 75% of random pairs" (percentile > 75). Uses the floor
    value computed during calibration (brief §3 step 4), which IS this exact
    measurement. percentile follows metric.calibrate()'s convention: higher
    percentile = closer/better (see FINDINGS.md for the sign-flip bug this
    once had).
    """
    floor_percentile = float((bundle.null_distribution > bundle.floor).mean() * 100.0)
    logger.info("floor=%.4f -> percentile=%.1f", bundle.floor, floor_percentile)
    assert floor_percentile > 75.0, (
        f"Same-track floor should be closer than 75% of the null distribution, "
        f"got {floor_percentile:.1f}th percentile"
    )


def test_music_vs_silence_large_distance(model, bundle, library_clips):
    """Proxy for "music vs. spoken-word podcast": music vs. silence should
    be a clearly large D_brain (percentile < 25, i.e. closer than only a
    small fraction of random pairs) — silence carries no musical structure
    at all, the starkest possible contrast available without fetching
    external speech audio.
    """
    music_clip = library_clips[0]
    silence_path = Path("data/clip_library/_baseline_silence.wav")
    if not silence_path.exists():
        ingest.generate_silence(silence_path)
    result = _score(model, bundle, music_clip, silence_path)
    percentile = float((bundle.null_distribution > result.D_brain).mean() * 100.0)
    logger.info("music_vs_silence: D_brain=%.4f percentile=%.1f", result.D_brain, percentile)
    assert percentile < 25.0, (
        f"Music vs. silence should score below the 25th percentile of null, got {percentile:.1f}th"
    )


def test_loudness_perturbation_stays_small(model, bundle, library_clips, tmp_path):
    """Loudness-perturbed copy of a clip (±6 LU) -> still small D_brain
    (percentile > 75). If this fails, loudness normalization
    (ingest.normalize_clip) is broken.
    """
    import soundfile as sf

    clip = library_clips[0]
    audio, sr = sf.read(str(clip))
    louder = np.clip(audio * (10 ** (6 / 20)), -1.0, 1.0)  # +6dB approx +6LU
    louder_path = tmp_path / "louder.wav"
    sf.write(str(louder_path), louder, sr, subtype="FLOAT")

    result = _score(model, bundle, clip, louder_path)
    percentile = float((bundle.null_distribution > result.D_brain).mean() * 100.0)
    logger.info("loudness_perturbed: D_brain=%.4f percentile=%.1f", result.D_brain, percentile)
    assert percentile > 75.0, (
        f"A loudness-perturbed copy should still score close (>75th percentile), got {percentile:.1f}th — "
        "check that inputs are being loudness-normalized before TRIBE sees them"
    )


def test_genre_discriminability(model, bundle, library_clips):
    """For a reference, a genre-matched clip must score closer than a
    genre-mismatched clip, across at least 5 reference tracks. This is the
    load-bearing test — if it fails, stop and iterate on masking/baseline/
    weights before building the optimizer on top of this metric.
    """
    # library clip filenames double as genre labels (see
    # scripts/build_clip_library.py GENRES) — grouped into two families below
    # as the best available proxy for genre-match without external labels.
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
        distances = [(_score(model, bundle, ref, c).D_brain, c) for c in others]
        distances.sort(key=lambda x: x[0])
        closest, farthest = distances[0], distances[-1]
        ref_fam = family(ref.stem)
        if ref_fam == "other":
            continue  # skip references without a clear family for this proxy test
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
        f"Genre-matched clips should usually score closer than mismatched ones; "
        f"got {correct}/{total} correct — metric may not be measuring musical content"
    )
