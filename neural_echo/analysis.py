"""Measured audio features Claude uses to build Daniel's initial plan."""

import librosa
import numpy as np


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
    # This deliberately avoids WhisperX/Gemini: the deployable pipeline uses
    # Claude plus local measured features and never sends reference audio to a
    # second multimodal model.
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
