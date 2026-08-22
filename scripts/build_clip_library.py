"""Builds the calibration clip library: generates diverse 45s reference clips
via ElevenLabs, normalizes them, and computes the calibration bundle used by
neural_echo.metric / neural_echo.optimizer.

This spends real ElevenLabs API credits (one call per clip in GENRES below,
plus one extra call for the same-track-floor pair). Run with --dry-run to
skip generation and reuse whatever mp3s are already in data/clip_library/raw/.

Usage: .venv/bin/python scripts/build_clip_library.py [--dry-run] [--n N]
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from neural_echo import calibration, compat, ingest
from neural_echo.generator import Chunk, DynamicArc, ElevenLabsGenerator, Genome

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/clip_library/raw")
NORM_DIR = Path("data/clip_library/normalized")

# 10 hand-authored genomes spanning genre/mood/instrumentation/tempo/dynamics —
# deliberately NOT LLM-generated: this is a fixed calibration set, not part of
# the optimizer loop, so it needs to be reproducible and diverse by design.
GENRES = [
    dict(name="deep_techno", bpm=124, key_mode="A minor", instrumentation=["synth bass", "hi-hats", "analog pads"],
         texture_density=0.6, dynamic_arc=DynamicArc.crescendo, vocal_presence=False, brightness=0.4, space_reverb=0.5,
         styles=["deep melodic techno", "driving bassline", "atmospheric pads", "124 BPM"], negative=["vocals"]),
    dict(name="orchestral_cinematic", bpm=90, key_mode="D minor", instrumentation=["strings", "brass", "timpani"],
         texture_density=0.8, dynamic_arc=DynamicArc.peak_and_fall, vocal_presence=False, brightness=0.5, space_reverb=0.7,
         styles=["epic orchestral", "cinematic strings", "sweeping brass", "dramatic timpani"], negative=["electronic", "vocals"]),
    dict(name="lofi_hiphop", bpm=82, key_mode="F major", instrumentation=["dusty piano", "vinyl crackle", "soft drums"],
         texture_density=0.35, dynamic_arc=DynamicArc.flat, vocal_presence=False, brightness=0.3, space_reverb=0.3,
         styles=["lo-fi hip-hop", "chill dusty piano loop", "vinyl crackle texture", "82 BPM", "relaxed"], negative=["vocals", "harsh"]),
    dict(name="acoustic_folk", bpm=100, key_mode="G major", instrumentation=["acoustic guitar", "fiddle", "light percussion"],
         texture_density=0.4, dynamic_arc=DynamicArc.multi_peak, vocal_presence=False, brightness=0.6, space_reverb=0.2,
         styles=["acoustic folk", "warm fingerpicked guitar", "fiddle melody", "100 BPM"], negative=["electronic", "vocals"]),
    dict(name="jazz_trio", bpm=120, key_mode="Bb major", instrumentation=["upright bass", "brushed drums", "piano"],
         texture_density=0.5, dynamic_arc=DynamicArc.multi_peak, vocal_presence=False, brightness=0.55, space_reverb=0.4,
         styles=["swing jazz trio", "walking upright bass", "brushed drums", "improvised piano"], negative=["vocals", "electronic"]),
    dict(name="rock_anthem", bpm=140, key_mode="E minor", instrumentation=["distorted guitar", "drums", "bass guitar"],
         texture_density=0.85, dynamic_arc=DynamicArc.crescendo, vocal_presence=False, brightness=0.7, space_reverb=0.3,
         styles=["driving rock anthem", "distorted power chords", "energetic drums", "140 BPM"], negative=["vocals"]),
    dict(name="ambient_drone", bpm=60, key_mode="C major", instrumentation=["synth drone", "field recordings", "soft pads"],
         texture_density=0.15, dynamic_arc=DynamicArc.flat, vocal_presence=False, brightness=0.25, space_reverb=0.9,
         styles=["ambient drone", "slow evolving pads", "field recording textures", "meditative"], negative=["drums", "vocals", "fast"]),
    dict(name="reggaeton", bpm=95, key_mode="A minor", instrumentation=["dembow drums", "synth bass", "percussion"],
         texture_density=0.7, dynamic_arc=DynamicArc.crescendo, vocal_presence=False, brightness=0.65, space_reverb=0.35,
         styles=["reggaeton instrumental", "dembow rhythm", "punchy synth bass", "95 BPM"], negative=["vocals"]),
    dict(name="classical_piano", bpm=70, key_mode="C minor", instrumentation=["solo piano"],
         texture_density=0.45, dynamic_arc=DynamicArc.peak_and_fall, vocal_presence=False, brightness=0.5, space_reverb=0.5,
         styles=["romantic solo piano", "expressive dynamics", "classical", "70 BPM"], negative=["drums", "electronic", "vocals"]),
    dict(name="drum_and_bass", bpm=174, key_mode="F minor", instrumentation=["breakbeat drums", "sub bass", "synth stabs"],
         texture_density=0.75, dynamic_arc=DynamicArc.multi_peak, vocal_presence=False, brightness=0.6, space_reverb=0.4,
         styles=["liquid drum and bass", "fast breakbeats", "deep sub bass", "174 BPM"], negative=["vocals"]),
]


def genome_for(g: dict) -> Genome:
    third = 15000
    return Genome(
        bpm=g["bpm"], key_mode=g["key_mode"], instrumentation=g["instrumentation"],
        texture_density=g["texture_density"], dynamic_arc=g["dynamic_arc"],
        vocal_presence=g["vocal_presence"], brightness=g["brightness"], space_reverb=g["space_reverb"],
        section_count=3,
        chunks=[
            Chunk(text="[Intro]", duration_ms=third, positive_styles=g["styles"], negative_styles=g["negative"]),
            Chunk(text="[Main]", duration_ms=third, positive_styles=g["styles"], negative_styles=g["negative"]),
            Chunk(text="[Outro]", duration_ms=third, positive_styles=g["styles"], negative_styles=g["negative"]),
        ],
        rationale=f"calibration clip: {g['name']}",
    )


async def generate_raw_clips(n: int, dry_run: bool) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = [g for g in GENRES[:n] if not (RAW_DIR / f"{g['name']}.mp3").exists()]
    already_done = [RAW_DIR / f"{g['name']}.mp3" for g in GENRES[:n] if (RAW_DIR / f"{g['name']}.mp3").exists()]
    if already_done:
        logger.info("Skipping %d already-generated clips: %s", len(already_done), [p.stem for p in already_done])
    if not todo:
        return already_done

    genomes = [genome_for(g) for g in todo]
    gen = ElevenLabsGenerator(output_dir=RAW_DIR, dry_run=dry_run)
    logger.info("Generating %d calibration clips via ElevenLabs (dry_run=%s)...", len(genomes), dry_run)
    results = await gen.generate_batch(genomes)
    paths = list(already_done)
    for name, r in zip([g["name"] for g in todo], results):
        if r.error:
            logger.error("Failed to generate %s: %s", name, r.error)
            continue
        named_path = RAW_DIR / f"{name}.mp3"
        Path(r.audio_path).rename(named_path)
        paths.append(named_path)
    return paths


def normalize_all(raw_paths: list[Path]) -> list[Path]:
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in raw_paths:
        out_path = NORM_DIR / f"{p.stem}.wav"
        ingest.normalize_clip(p, out_path, window_s=45.0, start_fraction=0.0)
        out.append(out_path)
    return out


def make_same_track_excerpts(raw_paths: list[Path]) -> list[tuple[Path, Path]]:
    """Two non-overlapping ~18s excerpts of EVERY library clip, for the noise
    floor measurement (brief §3 step 4). Averaging across all of them (see
    calibration.build_calibration) is far less noisy than a single track's
    measurement, which is what a 10-clip library needs to hit a stable floor.
    """
    import soundfile as sf

    pairs = []
    for i, src in enumerate(raw_paths):
        duration = sf.info(str(src)).duration
        if duration < 40:
            logger.warning("Clip %s too short for non-overlapping floor excerpts; skipping", src.stem)
            continue
        p1 = NORM_DIR / f"_floor_{i}_a.wav"
        p2 = NORM_DIR / f"_floor_{i}_b.wav"
        ingest.normalize_clip(src, p1, window_s=18.0, start_fraction=0.0)
        ingest.normalize_clip(src, p2, window_s=18.0, start_s=duration - 18.0)
        pairs.append((p1, p2))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="reuse existing mp3s in data/clip_library/raw, no API calls")
    parser.add_argument("--n", type=int, default=len(GENRES), help="number of calibration clips to generate")
    args = parser.parse_args()

    load_dotenv()

    if args.dry_run:
        raw_paths = sorted(RAW_DIR.glob("*.mp3"))
        if not raw_paths:
            logger.error("No existing clips found in %s for --dry-run", RAW_DIR)
            sys.exit(1)
    else:
        raw_paths = asyncio.run(generate_raw_clips(args.n, dry_run=False))

    logger.info("Normalizing %d clips to 45s / -14 LUFS / 44.1kHz...", len(raw_paths))
    norm_paths = normalize_all(raw_paths)

    same_track = make_same_track_excerpts(raw_paths)

    logger.info("Loading TRIBE model...")
    model = compat.get_tribe_model()

    logger.info("Building calibration bundle...")
    bundle = calibration.build_calibration(model, norm_paths, same_track_excerpts=same_track)
    bundle.save()

    logger.info(
        "Done. anatomical=%d data_driven=%d overlap=%d final_mask=%d null_median=%.4f floor=%.4f",
        bundle.anatomical_mask_size, bundle.data_driven_mask_size, bundle.overlap_size,
        bundle.vertex_mask.sum(), float(bundle.null_distribution.mean()), bundle.floor,
    )


if __name__ == "__main__":
    main()
