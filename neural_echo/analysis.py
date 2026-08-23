"""librosa audio analysis (for the optimizer's initial-plan prompt) and CLAP-
based constraint adherence / novelty scoring (brief §5).

CLAP is a hard filter, not a weighted term: the optimizer only scores
candidates whose adherence clears a calibrated threshold tau; a candidate
whose novelty (audio-audio similarity to the reference) exceeds the ceiling,
or whose tempo/instrumentation delta vs. the reference is too small, is a
"cover" and gets rejected outright — never blended into the brain-cost score.
"""
import logging
import threading

import librosa
import numpy as np

logger = logging.getLogger(__name__)

CLAP_MODEL_NAME = "laion/larger_clap_music_and_speech"
NOVELTY_AUDIO_SIM_CEILING = 0.95  # above this, candidate is judged a near-cover
MIN_TEMPO_DELTA_FRAC = 0.05        # or a >5% tempo change is required for "not a cover"

_clap_lock = threading.Lock()
_clap_model = None
_clap_processor = None


def _get_clap():
    global _clap_model, _clap_processor
    if _clap_model is not None:
        return _clap_model, _clap_processor
    with _clap_lock:
        if _clap_model is not None:
            return _clap_model, _clap_processor
        from transformers import ClapModel, ClapProcessor

        _clap_model = ClapModel.from_pretrained(CLAP_MODEL_NAME)
        _clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)
        _clap_model.eval()
        return _clap_model, _clap_processor


def analyze_reference(audio_path: str) -> dict:
    """librosa features for the optimizer's Generation-0 prompt (brief §4)."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_profile = chroma.mean(axis=1)
    key_idx = int(np.argmax(key_profile))
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_density = float(len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / (len(y) / sr))

    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.sum(harmonic ** 2))
    percussive_energy = float(np.sum(percussive ** 2))
    hp_ratio = harmonic_energy / (percussive_energy + 1e-9)

    # crude vocal-presence heuristic: harmonic energy concentrated in the
    # 200-4000Hz vocal formant range relative to total, on top of HPSS split.
    # (brief §5 lyrics-confound: also cross-check against WhisperX transcript
    # length in optimizer.py, which is a stronger signal when available.)
    stft = np.abs(librosa.stft(harmonic))
    freqs = librosa.fft_frequencies(sr=sr)
    vocal_band = (freqs >= 200) & (freqs <= 4000)
    vocal_energy_frac = float(stft[vocal_band].sum() / (stft.sum() + 1e-9))

    return {
        "tempo_bpm": tempo,
        "key_estimate": key_names[key_idx],
        "spectral_centroid_mean_hz": float(spectral_centroid.mean()),
        "rms_envelope_shape": {
            "mean": float(rms.mean()), "std": float(rms.std()),
            "max": float(rms.max()), "attack_frac": float(np.argmax(rms) / len(rms)),
        },
        "onset_density_per_s": onset_density,
        "harmonic_percussive_ratio": hp_ratio,
        "likely_has_vocals": vocal_energy_frac > 0.35,
        "vocal_band_energy_fraction": vocal_energy_frac,
        "duration_s": float(len(y) / sr),
    }


def clap_audio_embedding(audio_path: str) -> np.ndarray:
    import soundfile as sf

    model, processor = _get_clap()
    waveform, sr = sf.read(audio_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    target_sr = processor.feature_extractor.sampling_rate
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    # NOTE: transformers renamed ClapProcessor's `audios=` kwarg to `audio=`
    # at some point after this code was originally written — `audios=` now
    # raises ValueError instead of being silently accepted.
    inputs = processor(audio=waveform, sampling_rate=target_sr, return_tensors="pt")
    import torch
    with torch.no_grad():
        out = model.get_audio_features(**inputs)
    emb = out.pooler_output if hasattr(out, "pooler_output") else out
    return emb.squeeze(0).numpy()


def clap_text_embedding(text: str) -> np.ndarray:
    model, processor = _get_clap()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    import torch
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    emb = out.pooler_output if hasattr(out, "pooler_output") else out
    return emb.squeeze(0).numpy()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def constraint_adherence(candidate_audio_path: str, constraint_text: str) -> float:
    """Cosine similarity between candidate audio and the user's constraint
    text, in CLAP's joint embedding space. Compared against the optimizer's
    adherence_tau threshold (neural_echo/optimizer.py) — a bare cosine number
    means little on its own."""
    audio_emb = clap_audio_embedding(candidate_audio_path)
    text_emb = clap_text_embedding(constraint_text)
    return _cosine(audio_emb, text_emb)


def novelty_check(
    candidate_audio_path: str,
    reference_audio_path: str,
    reference_analysis: dict,
    reference_embedding: np.ndarray | None = None,
) -> dict:
    """Hard novelty filter (brief §5): audio-audio CLAP similarity below a
    ceiling AND a minimum tempo delta vs. the reference. Either condition
    failing marks the candidate a near-cover — rejected outright, reported
    to the LLM as a signal, never blended into the brain-cost score.

    Pass reference_embedding (from a single upfront clap_audio_embedding
    call) when checking many candidates against one fixed reference — e.g.
    the optimizer loop — to avoid re-embedding the reference every call.
    """
    cand_emb = clap_audio_embedding(candidate_audio_path)
    ref_emb = reference_embedding if reference_embedding is not None else clap_audio_embedding(reference_audio_path)
    audio_sim = _cosine(cand_emb, ref_emb)

    y, sr = librosa.load(candidate_audio_path, sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    cand_tempo = float(np.atleast_1d(tempo)[0])
    ref_tempo = reference_analysis["tempo_bpm"]
    tempo_delta_frac = abs(cand_tempo - ref_tempo) / max(ref_tempo, 1e-6)

    is_near_cover = (audio_sim > NOVELTY_AUDIO_SIM_CEILING) and (tempo_delta_frac < MIN_TEMPO_DELTA_FRAC)

    return {
        "audio_similarity": audio_sim,
        "candidate_tempo_bpm": cand_tempo,
        "tempo_delta_frac": tempo_delta_frac,
        "is_near_cover": is_near_cover,
    }
