"""Turn a YouTube URL or uploaded file into a normalized 90s reference clip.

Normalization matters more than it looks: loudness and sample rate
differences would otherwise dominate TRIBE's predictions before any musical
content does, which would swamp metric.py's region-by-region comparison.
"""
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

TARGET_LUFS = -14.0
TARGET_SR = 44100
DEFAULT_WINDOW_S = 90.0
DEFAULT_START_FRACTION = 0.0  # Daniel's benchmark uses the track's first 90 seconds


def _youtube_cookies_path() -> Path | None:
    candidates = [
        os.environ.get("YOUTUBE_COOKIES_PATH"),
        "cookies.txt",
        "/app/data/cookies.txt",
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file() and path.stat().st_size > 0:
                return path
    return None


@dataclass
class NormalizedClip:
    path: Path
    duration_s: float
    source_duration_s: float
    start_s: float


def download_youtube_audio(url: str, out_dir: Path) -> Path:
    """Download best-effort audio via yt-dlp.

    NOTE (ToS): downloading audio from YouTube violates YouTube's Terms of
    Service. This path exists as a demo affordance only — the file-upload path
    below is the ToS-clean primary route. Surface this in the UI, not just here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(id)s.%(ext)s")
    # Invoke via `python -m yt_dlp` rather than the bare `yt-dlp` binary — a
    # subprocess doesn't inherit the venv's bin/ on PATH just because the
    # calling process is `.venv/bin/python`, so the bare command silently
    # fails with FileNotFoundError in exactly this setup (reproduced both
    # locally and on Render). sys.executable guarantees the same venv.
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", out_template,
        "--no-playlist",
        "--extractor-args", "youtube:player_client=android,ios,web",
    ]
    cookies_path = _youtube_cookies_path()
    if cookies_path:
        cmd.extend(["--cookies", str(cookies_path)])
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")
    candidates = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("yt-dlp reported success but produced no .wav file")
    return candidates[-1]


def normalize_clip(
    src_path: Path,
    out_path: Path,
    window_s: float = DEFAULT_WINDOW_S,
    start_fraction: float = DEFAULT_START_FRACTION,
    start_s: float | None = None,
) -> NormalizedClip:
    """Extract a fixed window, loudness-normalize to TARGET_LUFS, resample to
    TARGET_SR, and write mono float32 wav. This is the ONLY place raw source
    audio should be read — every downstream consumer (TRIBE and librosa)
    reads the output of this function so they all see identical stimuli.
    """
    info = sf.info(str(src_path))
    source_duration_s = info.duration

    if start_s is None:
        start_s = max(0.0, source_duration_s * start_fraction)
    actual_window_s = min(window_s, max(0.0, source_duration_s - start_s))
    if actual_window_s <= 0:
        start_s = 0.0
        actual_window_s = min(window_s, source_duration_s)

    audio, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono

    start_sample = int(start_s * sr)
    end_sample = int((start_s + actual_window_s) * sr)
    audio = audio[start_sample:end_sample]

    if sr != TARGET_SR:
        # simple resample via linear interpolation is good enough for TRIBE's
        # audio encoder (it resamples internally too); avoid pulling in a heavy
        # resampling dependency for this.
        n_target = round(len(audio) * TARGET_SR / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_target),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        sr = TARGET_SR

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if np.isfinite(loudness):
        audio = pyln.normalize.loudness(audio, loudness, TARGET_LUFS)
    audio = np.clip(audio, -1.0, 1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr, subtype="FLOAT")

    return NormalizedClip(
        path=out_path,
        duration_s=len(audio) / sr,
        source_duration_s=source_duration_s,
        start_s=start_s,
    )


def generate_silence(out_path: Path, duration_s: float = DEFAULT_WINDOW_S, sr: int = TARGET_SR) -> Path:
    audio = np.zeros(int(duration_s * sr), dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr, subtype="FLOAT")
    return out_path
