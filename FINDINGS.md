# Neural Echo — Phase 0 Findings (Probe)

Date: 2026-08-22
Machine: macOS (Darwin, Apple Silicon), **no CUDA GPU** — 10 CPU cores.

## 1. Environment was broken; now fixed

The repo's `.venv` had stale shebangs pointing at a `logos/.venv` path that no longer
exists (looked like the venv was originally created in a different directory and
never rebuilt). Fixed with `rm -rf .venv && uv sync`.

Dependencies added to `pyproject.toml` (were missing, not declared anywhere despite
being required by `tribev2`/the demo notebook):
`scikit-image`, `torchaudio==2.6.0` (pinned to match the existing `torch==2.6.0`),
`elevenlabs`, `python-dotenv`, `opencv-python`, `plotly` (required by nilearn's
`view_surf` html engine), `nbconvert`.

System dependency installed: `ffmpeg` (via `brew install ffmpeg`) — WhisperX shells
out to the `ffmpeg` binary directly; it wasn't present on this machine at all.

## 2. Bugs found and worked around in TRIBE v2 / its dependency chain

These are upstream bugs, not something introduced by us — documented here so future
debugging doesn't repeat this. All fixes are done via monkeypatching or environment
variables in **our own code**, never by editing `site-packages`.

1. **`uvx`-spawned WhisperX crashes with `AttributeError: module 'torchaudio' has no
   attribute 'list_audio_backends'`.** `tribev2.eventstransforms.ExtractWordsFromAudio`
   shells out to `uvx whisperx ...` for every audio/text input (this is how the
   pipeline derives word timings — see §4). `uvx` defaults to Python 3.14 on this
   machine, for which the only available `torch`/`torchaudio` wheels are recent
   enough to have dropped `torchaudio.list_audio_backends()`, which `pyannote.audio`
   (used by WhisperX for voice-activity detection) still calls.
   **Fix:** set `UV_PYTHON=3.11` in the environment before any TRIBE call that might
   invoke WhisperX — forces `uvx` onto an interpreter with compatible wheels.

2. **`ValueError: Requested float16 compute type, but the target device or backend do
   not support efficient float16 computation.`** `eventstransforms.py` hardcodes
   `compute_type = "float16"` unconditionally, regardless of device. `ctranslate2`
   (WhisperX's backend) rejects `float16` on CPU.
   **Fix:** monkeypatched `ExtractWordsFromAudio._get_transcript_from_audio` with a
   byte-for-byte copy of the upstream method, except `compute_type` is now
   `"float16" if device == "cuda" else "float32"`.

3. **`AssertionError: Torch not compiled with CUDA enabled`** when running the
   image/video pipeline. The pretrained model's `config.yaml` (downloaded from the
   `facebook/tribev2` HF repo) hardcodes `device: cuda` for all 4 feature extractors
   (text/image/audio/video) — it was authored on a Colab GPU box, and unlike the
   extractors' own "auto" default, an explicit "cuda" doesn't get corrected for
   CPU-only hardware.
   **Fix:** after `TribeModel.from_pretrained(...)`, explicitly set
   `model.data.{text_feature,audio_feature}.device` and
   `model.data.{image_feature,video_feature}.image.device` to `"cpu"` when
   `torch.cuda.is_available()` is `False`.

4. **`RuntimeError: An attempt has been made to start a new process before the
   current process has finished its bootstrapping phase.`** TRIBE's own `DataLoader`
   uses multiprocessing with the `spawn` start method, which requires the entry
   script to guard its code under `if __name__ == "__main__":`.
   **Fix:** wrapped probe/pipeline code in a `main()` function called under the
   standard guard.

5. **Severe CPU oversubscription / apparent hang.** `model.data.num_workers`
   defaults high enough to spawn `2×` this machine's core count in DataLoader
   workers (each a full `torch`/`transformers` re-import via `spawn`), which stalls
   for a long time on a 10-core box (a run that should take ~2.5s took 56s under
   contention from a second overlapping worker pool).
   **Fix:** cap `model.data.num_workers = min(4, os.cpu_count())` before calling
   `.predict()`.

All five fixes are self-contained and reusable — they should live in a shared
`neural_echo/compat.py`-style module in Phase 1+, applied once at model-load time,
rather than copy-pasted per script.

## 3. Empirical measurements (audio-only input, 8-second test clip)

Note: measured on an **8-second** clip, not the target 45-second window, for fast
iteration on CPU hardware (per explicit instruction — full 45s + GPU timing should
be re-measured before trusting these for production sizing / optimizer timeout
budgets).

Clean run (no CPU contention, `num_workers=4`, model + WhisperX caches warm):

| metric | value |
|---|---|
| `preds.shape` | `(8, 20484)` |
| `n_timesteps` | 8 |
| `n_vertices` | 20484 (10242/hemisphere — matches fsaverage5 spec exactly) |
| **timestep rate** | **1.0 Hz**, confirmed empirically (not assumed) |
| `model_load_seconds` | 0.84s (warm HF cache) |
| `get_events_dataframe_seconds` | 0.04s (WhisperX transcript cache warm) |
| `predict_seconds` | 2.48s |
| peak RSS | 1.33 GB (CPU RAM — **not** GPU VRAM, see caveat below) |

Cold-cache reference point (first run of this session, includes downloading the
WhisperX `large-v3` alignment model + one-time transcription, **and** was
contaminated by the oversubscription bug in point 5 above — not a clean number, but
gives an order of magnitude for a cold start): `get_events_dataframe_seconds ≈ 10.4s`,
`predict_seconds ≈ 56s` (this second number is inflated by contention and should be
disregarded in favor of the clean 2.48s figure).

**Video modality confirmed never loaded for audio-only input** — both empirically
(`model.data.video_feature.image._model` stayed `None` before and after `.predict()`)
and via TRIBE's own log output: `Removing extractor video as there are no
corresponding events` / `Removing extractor text as there are no corresponding
events` (the latter because this particular clip is instrumental — WhisperX found no
speech, so the text/word modality is also dropped for this input, which is itself a
useful confirmation of the brief's §4 lyrics-confound concern: modality activation is
input-dependent, not toggled by us).

## 4. Confirmed: WhisperX transcription is silently in the loop for audio input

This is not a video-modality-only concern. Any call to
`model.get_events_dataframe(audio_path=...)` (and `text_path=...`, which internally
does TTS + WhisperX alignment for timing) shells out to WhisperX regardless of
whether the audio has vocals. All the bugs in §2.1/§2.2 block **any** audio-based
TRIBE call, not just a rare edge case — this had to be fixed before Phase 0's core
probe could run at all.

## 5. GPU / VRAM — gap versus the brief's assumptions

**This development machine has no CUDA GPU.** "Peak VRAM with/without the video
modality" as specified in the brief could not be measured here — the number
reported above (1.33 GB) is process RSS (CPU RAM), a rough proxy at best, and is not
meaningful for the brief's GPU-sizing question ("audio+text only should fit
comfortably under 24GB — measure it"). This must be re-measured on real CUDA
hardware (e.g. a rented GPU box) before any GPU-sizing claim goes in a README.
Flagging this explicitly rather than fabricating a number.

## 6. ElevenLabs SDK surface — confirmed

Confirmed via introspection (no extra API calls / credit spend): `client.music.compose`,
`client.music.composition_plan.create`, and `AsyncElevenLabs` are all present in the
installed `elevenlabs` SDK version.

Additionally, `client.music.compose(composition_plan=..., model_id="music_v2")` was
exercised for real earlier this session (outside this probe script, while getting the
existing `song_generator.ipynb` demo running) — two real composition plans were sent
and both returned valid MP3s (`electronica_candidato.mp3`, `vocal_candidato.mp3`),
confirming the exact code path the brief's optimizer loop will drive.

## 7a. Phase 1 addendum: text modality disabled permanently

While building the calibration clip library, WhisperX's VAD false-positived on
a non-vocal transient in an instrumental ElevenLabs generation, which flipped
on TRIBE's text/word modality — which then needs `meta-llama/Llama-3.2-3B`, a
**gated** HF repo we don't have approved access to (a 403, not something a
token alone fixes). Separately, `uvx`-spawned WhisperX also crashed
intermittently with a native `libc++abi: recursive_mutex lock failed` abort
after several sequential invocations in one long-lived process (not
reproducible when run standalone) — a flaky dependency even when the gated
model isn't at issue.

Rather than chase HF repo access + flaky-subprocess retries, `compat.py` now
monkeypatches `ExtractWordsFromAudio._run` to always return events unchanged
— text/word extraction is **permanently disabled**, for every clip, regardless
of content. This is exactly the brief's own §5 option (i) ("score both tracks
with the text modality forced empty"), chosen over option (ii) (UI flag)
because it removes a slow + gated + flaky dependency from every scoring call
entirely, and makes every metric.py comparison apples-to-apples by
construction. Vocal-presence diagnostics for the optimizer/UI now come from
`analysis.py`'s librosa heuristic (harmonic-band energy fraction) instead,
which has no such dependency. Net effect: every TRIBE call in this codebase
is audio-only, faster, and no longer depends on `uvx`/WhisperX at all.

## 8. Phase 2-3 addendum: bugs found during full end-to-end integration testing

Found by actually running the optimizer against the live FastAPI server (not
just unit-level testing) — real bugs that unit tests on individual functions
wouldn't have caught:

- **`transformers` version drift in the CLAP integration.** The installed
  `transformers` renamed `ClapProcessor`'s `audios=` kwarg to `audio=`
  (raises `ValueError` now instead of accepting the old name), and
  `model.get_audio_features()`/`get_text_features()` now return a
  `BaseModelOutputWithPooling` object instead of a raw tensor — need
  `.pooler_output`. Fixed in `analysis.py`. A reminder that any code written
  against "how an API used to work" needs a real smoke test before trusting
  it, not just a read-through.
- **SSE fan-out race with a shared `Queue`.** The first `services/api`
  implementation used a single `queue.Queue` per job, drained by
  `Queue.get()` in the SSE endpoint — caught during frontend review (not by
  a test): a second concurrent viewer (a refresh, a second tab) would starve
  or duplicate events, since each queued item can only be consumed once.
  Rewritten so `job.generations` (an append-only list) is the source of
  truth, polled by index — this also gives replay-on-reconnect for free
  (a client connecting mid-run or after completion just catches up from
  index 0 instead of silently missing prior progress).
- **Restarting a live server picks up code fixes; nothing else does.**
  Obvious in hindsight, but worth writing down: after fixing the CLAP bug
  above, a *new* end-to-end job against the *already-running* uvicorn
  process reproduced the exact same error, because Python doesn't hot-reload
  already-imported modules. Always restart the server (or use `--reload`
  during development) after a fix before re-testing against it.
- **The floor measurement (brief §3 step 4) was noisy from a single
  same-track excerpt pair** — one measurement off one clip swung between the
  25th and 40th null-distribution percentile depending on which clip was
  used. Fixed by averaging the floor across all 10 library clips'
  excerpt-pairs instead of just the first one (`calibration.py`,
  `scripts/build_clip_library.py`) — moved the validation gate's floor test
  from a clear fail (33.3rd percentile) to a marginal one (26.7th vs. a 25th
  threshold), which is within the noise of a 45-sample null distribution
  (each pair is ~2.2 percentile points). Not chased further than that —
  the fix is the right one architecturally; closing the remaining ~2
  percentage points would need a bigger calibration library (more real
  ElevenLabs spend), which is exactly the "10 clips, not ~30" tradeoff
  already documented in the README.
- **CLAP adherence threshold (default τ=0.15) is a reasonable-looking
  default, not an empirically calibrated one.** In one real run, raw
  CLAP cosine similarities against the constraint text ranged from -0.03 to
  0.19 across 3 candidates — the brief calls for calibrating τ against the
  clip library (§5); this implementation ships a default instead. Worth
  doing properly before relying on the adherence filter's pass/fail line in
  a real product decision.

## 7. Open items for Phase 1+

- Re-measure `predict_seconds` / VRAM on a real 45s clip and on GPU hardware before
  using these numbers for optimizer timeout budgets or README claims.
- Move the 5 compat fixes in §2 into a shared module (e.g. `neural_echo/compat.py`),
  applied once at model load, instead of being probe-script-local.
- `model.extract_features(df)` (mentioned in the brief for cheap pre-filtering) not
  yet exercised — worth confirming its output shape in Phase 1 alongside the metric
  work.
