"""Phase 0 probe for Neural Echo: environment sanity + TRIBE v2 timing/shape checks.

Run with: .venv/bin/python scripts/phase0_probe.py
"""
import json
import os
import resource
import time

os.environ.setdefault("UV_PYTHON", "3.11")  # forces uvx-spawned whisperx onto a python
# with working torch/torchaudio wheels (uvx's default python (3.14) only has torch/torchaudio
# wheels newer than the one that dropped torchaudio.list_audio_backends(), which pyannote.audio
# (a whisperx dependency, used for VAD) still calls -> AttributeError without this).

AUDIO_PATH = "short_test_clip.mp3"


# tribev2.eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio hardcodes
# compute_type="float16" for the whisperx subprocess regardless of device, which
# ctranslate2/faster-whisper rejects on CPU ("Requested float16 compute type, but the
# target device or backend do not support efficient float16 computation."). Patch in a
# device-aware compute_type; everything else is copied verbatim from upstream.
def _patch_whisperx_compute_type():
    import subprocess
    import tempfile
    from pathlib import Path

    import pandas as pd
    import torch
    from tribev2 import eventstransforms

    def _get_transcript_from_audio(wav_filename, language):
        language_codes = dict(english="en", french="fr", spanish="es", dutch="nl", chinese="zh")
        if language not in language_codes:
            raise ValueError(f"Language {language} not supported")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"

        with tempfile.TemporaryDirectory() as output_dir:
            cmd = [
                "uvx", "whisperx", str(wav_filename),
                "--model", "large-v3",
                "--language", language_codes[language],
                "--device", device,
                "--compute_type", compute_type,
                "--batch_size", "16",
                "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                "--output_dir", output_dir,
                "--output_format", "json",
            ]
            cmd = [c for c in cmd if c]
            env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"whisperx failed:\n{result.stderr}")

            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript = json.loads(json_path.read_text())

        words = []
        for i, segment in enumerate(transcript["segments"]):
            sentence = segment["text"].replace('"', "")
            for word in segment["words"]:
                if "start" not in word:
                    continue
                words.append({
                    "text": word["word"].replace('"', ""),
                    "start": word["start"],
                    "duration": word["end"] - word["start"],
                    "sequence_id": i,
                    "sentence": sentence,
                })
        return pd.DataFrame(words)

    eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
        _get_transcript_from_audio
    )


def peak_rss_gb():
    import platform
    # ru_maxrss is bytes on macOS, KB on Linux
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 ** 3) if platform.system() == "Darwin" else raw / (1024 ** 2)


def main():
    import soundfile as sf
    from dotenv import load_dotenv
    from elevenlabs.client import ElevenLabs
    from tribev2 import TribeModel

    load_dotenv()
    _patch_whisperx_compute_type()

    findings = {}

    # --- Load model ---
    t0 = time.time()
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
    findings["model_load_seconds"] = round(time.time() - t0, 2)

    # The pretrained config (config.yaml in the HF snapshot) hardcodes device: cuda for all
    # 4 feature extractors (text/image/audio/video), presumably authored on a Colab GPU box.
    # On CPU-only hardware this makes any .to(device) call inside HF transformers try to
    # lazy-init CUDA and raise `AssertionError: Torch not compiled with CUDA enabled`.
    # Patch to cpu explicitly rather than relying on the extractors' own "auto" resolution,
    # since the pretrained config sets an explicit "cuda" (not "auto").
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    findings["cuda_available"] = has_cuda
    if not has_cuda:
        model.data.text_feature.device = "cpu"
        model.data.image_feature.image.device = "cpu"
        model.data.audio_feature.device = "cpu"
        model.data.video_feature.image.device = "cpu"

    # model.data.num_workers defaults high enough to spawn 2x the machine's core count
    # in DataLoader workers (each a full torch/transformers re-import via spawn), which
    # oversubscribes CPU and stalls indefinitely on a small dev box. Cap it.
    import os as _os
    model.data.num_workers = min(4, _os.cpu_count() or 4)
    findings["num_workers"] = model.data.num_workers

    # --- Confirm audio-only input never loads the video encoder ---
    video_model_loaded_before = getattr(model.data.video_feature.image, "_model", None) is not None
    findings["video_model_loaded_before_predict"] = video_model_loaded_before

    # --- Run TRIBE on the local test clip (audio-only path) ---
    duration_s = sf.info(AUDIO_PATH).duration
    findings["input_audio_duration_s"] = round(duration_s, 3)

    t0 = time.time()
    df = model.get_events_dataframe(audio_path=AUDIO_PATH)
    findings["get_events_dataframe_seconds"] = round(time.time() - t0, 2)

    t0 = time.time()
    preds, segments = model.predict(events=df)
    findings["predict_seconds"] = round(time.time() - t0, 2)

    findings["preds_shape"] = list(preds.shape)
    findings["n_timesteps"] = int(preds.shape[0])
    findings["n_vertices"] = int(preds.shape[1])
    findings["timestep_rate_hz"] = round(preds.shape[0] / duration_s, 4)

    video_model_loaded_after = getattr(model.data.video_feature.image, "_model", None) is not None
    findings["video_model_loaded_after_audio_predict"] = video_model_loaded_after

    findings["peak_rss_gb_note"] = "CPU RAM proxy, not GPU VRAM (no CUDA device on this machine)"
    findings["peak_rss_gb"] = round(peak_rss_gb(), 2)

    # --- Confirm ElevenLabs SDK surface (introspection only, no API calls / no credit spend) ---
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    findings["elevenlabs_music_compose_exists"] = hasattr(client.music, "compose")
    findings["elevenlabs_composition_plan_create_exists"] = hasattr(
        client.music, "composition_plan"
    ) and hasattr(client.music.composition_plan, "create")
    try:
        from elevenlabs.client import AsyncElevenLabs  # noqa: F401
        findings["async_client_available"] = True
    except ImportError:
        findings["async_client_available"] = False

    print(json.dumps(findings, indent=2))
    with open("phase0_probe_result.json", "w") as f:
        json.dump(findings, f, indent=2)


if __name__ == "__main__":
    main()
