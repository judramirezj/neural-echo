"""Compatibility shims for running TRIBE v2 on CPU-only / non-Colab hardware.

These are workarounds for upstream bugs in `tribev2` / its dependency chain,
found and documented in FINDINGS.md (Phase 0). None of this edits site-packages;
everything is monkeypatched or set via environment variables at import/load time
so it stays reproducible across machines and easy to remove once upstream fixes
ship. Import `get_tribe_model()` to get a working, CPU-safe, warm model instance.
"""
import gc
import os
import threading

os.environ.setdefault("UV_PYTHON", "3.11")  # harmless if whisperx is never invoked
# (see _patch_disable_text_modality below) — kept as a defensive default in case
# any other code path still shells out to uvx-managed tools.

_model_lock = threading.Lock()
_model = None
_patched = False


def _patch_disable_text_modality():
    """Force the text/word modality permanently empty (brief §5, option i:
    "score both tracks with the text modality forced empty").

    Rationale, not just a workaround: tribev2's audio pipeline silently shells
    out to WhisperX (`uvx whisperx ...`, large-v3) on every audio input to
    derive word timings for the text modality — even for purely instrumental
    ElevenLabs generations, where WhisperX's VAD occasionally false-positives
    on non-vocal transients. When it does, TRIBE's text extractor then needs
    `meta-llama/Llama-3.2-3B`, which is a gated HF repo we don't have approved
    access to, and even when it works, WhisperX crashed intermittently on this
    machine with a native "recursive_mutex lock failed" abort after several
    sequential invocations (not reproducible standalone; see FINDINGS.md).
    Disabling text extraction outright removes a slow, flaky, and gated
    dependency from every scoring call, and makes every comparison in
    metric.py apples-to-apples: reference and candidate are always scored on
    audio content alone, never on incidental lyrics content. Vocal presence
    for the UI/optimizer diagnostics is instead estimated in analysis.py via
    librosa heuristics, which has no such dependency.
    """
    from tribev2 import eventstransforms

    def _run(self, events):
        return events

    eventstransforms.ExtractWordsFromAudio._run = _run


def apply_patches():
    global _patched
    if _patched:
        return
    _patch_disable_text_modality()
    _patched = True


def get_tribe_model(cache_folder: str | None = None):
    """Return a warm, process-wide singleton TribeModel, patched for CPU use.

    Call once per process (e.g. at FastAPI startup) — TRIBE inference is the
    latency bottleneck and reloading the model per request is a correctness
    hazard for a warm-worker design (see project brief §4 Throughput).

    cache_folder defaults to $TRIBE_CACHE_DIR (falling back to "./cache") so
    deployments can point it at a persistent disk — on Render, the default
    HuggingFace hub cache (~/.cache/huggingface) and this folder both live
    outside the mounted disk and would otherwise re-download the model on
    every restart. See render.yaml, which sets both HF_HOME and
    TRIBE_CACHE_DIR under the mounted /app/data disk.
    """
    if cache_folder is None:
        cache_folder = os.environ.get("TRIBE_CACHE_DIR", "./cache")
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model

        apply_patches()
        import torch
        from tribev2 import TribeModel

        model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_folder)

        if not torch.cuda.is_available():
            # tribev2's pretrained config.yaml hardcodes device: cuda for all 4
            # feature extractors (authored on a Colab GPU box) — unlike the
            # extractors' own "auto" default, an explicit "cuda" isn't corrected
            # for CPU-only hardware and raises AssertionError on first use.
            model.data.text_feature.device = "cpu"
            model.data.image_feature.image.device = "cpu"
            model.data.audio_feature.device = "cpu"
            model.data.video_feature.image.device = "cpu"

        # Default num_workers spawns ~2x this machine's core count in DataLoader
        # workers (each a full torch/transformers re-import via spawn), which
        # oversubscribes CPU and stalls indefinitely on small boxes.
        model.data.num_workers = min(4, os.cpu_count() or 4)

        _model = model
        return _model


def has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def release_inference_memory():
    """Release intermediate TRIBE tensors while keeping the warm model loaded.

    Daniel's optimizer does this after every prediction to prevent gradual VRAM
    fragmentation and OOMs during long single-lineage runs.
    """
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
