"""Builds and persists the calibration bundle: baseline, vertex mask, dynamics
z-score stats, null distribution, and noise floor (brief §3 steps 0b/1/4).

Run once via `python -m neural_echo.calibration build` (or scripts/build_calibration.py).
Everything downstream (metric.distance, the optimizer) loads the cached bundle
instead of recomputing it per request.
"""
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import atlases, metric

logger = logging.getLogger(__name__)

CALIBRATION_DIR = Path("data/clip_library")
BUNDLE_PATH = CALIBRATION_DIR / "calibration_bundle.npz"
DATA_DRIVEN_TOP_K = 4000


@dataclass
class CalibrationBundle:
    baseline_mean: np.ndarray          # (20484,)
    vertex_mask: np.ndarray            # (20484,) bool — anatomical ∩ data-driven
    network_labels: np.ndarray | None  # (20484,) int, or None
    dynamics_mean: np.ndarray          # (dynamics_raw_dim,)
    dynamics_std: np.ndarray           # (dynamics_raw_dim,)
    null_distribution: np.ndarray      # (n_pairs,) D_brain over random clip pairs
    floor: float                       # D_brain between 2 excerpts of the same track
    anatomical_mask_size: int
    data_driven_mask_size: int
    overlap_size: int

    def save(self, path: Path = BUNDLE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            baseline_mean=self.baseline_mean,
            vertex_mask=self.vertex_mask,
            network_labels=(self.network_labels if self.network_labels is not None else np.array([])),
            dynamics_mean=self.dynamics_mean,
            dynamics_std=self.dynamics_std,
            null_distribution=self.null_distribution,
            floor=np.array([self.floor]),
            anatomical_mask_size=np.array([self.anatomical_mask_size]),
            data_driven_mask_size=np.array([self.data_driven_mask_size]),
            overlap_size=np.array([self.overlap_size]),
        )
        logger.info("Saved calibration bundle to %s", path)

    @classmethod
    def load(cls, path: Path = BUNDLE_PATH) -> "CalibrationBundle":
        data = np.load(path, allow_pickle=False)
        network_labels = data["network_labels"]
        return cls(
            baseline_mean=data["baseline_mean"],
            vertex_mask=data["vertex_mask"].astype(bool),
            network_labels=(network_labels if network_labels.size else None),
            dynamics_mean=data["dynamics_mean"],
            dynamics_std=data["dynamics_std"],
            null_distribution=data["null_distribution"],
            floor=float(data["floor"][0]),
            anatomical_mask_size=int(data["anatomical_mask_size"][0]),
            data_driven_mask_size=int(data["data_driven_mask_size"][0]),
            overlap_size=int(data["overlap_size"][0]),
        )


def compute_baseline_mean(model, cache_dir: Path) -> np.ndarray:
    """Mean TRIBE prediction over 45s of pink noise + 45s of silence (brief
    §3 step 0b) — the "any audio" constant prior to subtract from everything.
    """
    from . import ingest

    cache_dir.mkdir(parents=True, exist_ok=True)
    silence_path = cache_dir / "_baseline_silence.wav"
    noise_path = cache_dir / "_baseline_pinknoise.wav"
    if not silence_path.exists():
        ingest.generate_silence(silence_path)
    if not noise_path.exists():
        ingest.generate_pink_noise(noise_path)

    means = []
    for path in (silence_path, noise_path):
        df = model.get_events_dataframe(audio_path=str(path))
        preds, _ = model.predict(events=df)
        means.append(preds.mean(axis=0))
    return np.mean(means, axis=0)


def _raw_delta_for_clip(model, clip_path: Path, baseline_mean: np.ndarray) -> np.ndarray:
    df = model.get_events_dataframe(audio_path=str(clip_path))
    preds, _ = model.predict(events=df)
    trimmed = metric.trim_edges(preds)
    return metric.subtract_baseline(trimmed, baseline_mean)  # (T, 20484), unmasked


def build_data_driven_mask(deltas: list[np.ndarray], anatomical_mask: np.ndarray, top_k: int = DATA_DRIVEN_TOP_K) -> np.ndarray:
    """Empirical sensitivity map: per-vertex variance of ΔX across a diverse
    clip library, intersected with the anatomical mask (brief §3 step 1).
    """
    per_clip_means = [d.mean(axis=0) for d in deltas]  # each (20484,) — one point per clip
    stacked = np.stack(per_clip_means)  # (n_clips, 20484)
    variance = stacked.var(axis=0)
    top_k = min(top_k, variance.shape[0])
    top_idx = np.argsort(variance)[-top_k:]
    data_driven = np.zeros_like(anatomical_mask)
    data_driven[top_idx] = True
    return data_driven


def build_calibration(
    model,
    clip_paths: list[Path],
    same_track_excerpts: tuple[Path, Path] | list[tuple[Path, Path]] | None = None,
    top_k: int = DATA_DRIVEN_TOP_K,
    weights: dict = metric.DEFAULT_WEIGHTS,
) -> CalibrationBundle:
    if len(clip_paths) < 4:
        raise ValueError("Need at least 4 diverse clips to build a meaningful calibration bundle")

    cache_dir = CALIBRATION_DIR
    baseline_mean = compute_baseline_mean(model, cache_dir)

    logger.info("Running TRIBE on %d calibration clips...", len(clip_paths))
    deltas = [_raw_delta_for_clip(model, p, baseline_mean) for p in clip_paths]

    anatomical_mask = atlases.fetch_anatomical_mask()
    network_labels = atlases.fetch_yeo7_network_labels()

    data_driven_mask = build_data_driven_mask(deltas, anatomical_mask, top_k=top_k)
    vertex_mask = anatomical_mask & data_driven_mask
    overlap_size = int((anatomical_mask & data_driven_mask).sum())
    logger.info(
        "anatomical=%d data_driven=%d overlap=%d final_mask=%d",
        anatomical_mask.sum(), data_driven_mask.sum(), overlap_size, vertex_mask.sum(),
    )
    if vertex_mask.sum() < 50:
        raise RuntimeError(
            f"Final vertex mask only has {vertex_mask.sum()} vertices — anatomical/data-driven "
            "masks barely overlap. Something is wrong (see brief §3 step 1); investigate before "
            "using this calibration bundle."
        )

    profiles = []
    for delta in deltas:
        delta_masked = metric.apply_mask(delta, vertex_mask)
        roi_tc = metric._roi_mean_timecourse(delta_masked)
        network_labels_masked = network_labels[vertex_mask] if network_labels is not None else None
        profiles.append(metric.BrainProfile(
            spatial=delta_masked.mean(axis=0),
            vertex_std=delta_masked.std(axis=0),
            spectrum=metric._low_freq_spectrum(roi_tc, metric.TIMESTEP_RATE_HZ),
            autocorr=metric._autocorr_lags(roi_tc),
            network_corr_upper=metric._network_correlation_upper(delta_masked, network_labels_masked),
            n_timesteps=delta_masked.shape[0],
        ))

    dynamics_stack = np.stack([p.dynamics_raw for p in profiles])
    dynamics_mean = dynamics_stack.mean(axis=0)
    dynamics_std = dynamics_stack.std(axis=0)
    dynamics_std = np.where(dynamics_std < 1e-9, 1.0, dynamics_std)

    null_values = []
    for a, b in itertools.combinations(range(len(profiles)), 2):
        result = metric.distance(profiles[a], profiles[b], dynamics_mean, dynamics_std, weights)
        null_values.append(result.D_brain)
    null_distribution = np.array(null_values)
    logger.info(
        "Null distribution: n=%d mean=%.4f median=%.4f std=%.4f",
        len(null_distribution), null_distribution.mean(), np.median(null_distribution), null_distribution.std(),
    )

    if same_track_excerpts:
        # A floor measured from a single track's excerpt pair is noisy with a
        # small library (one sample). Average across as many same-track pairs
        # as are provided — much more robust, and costs nothing extra beyond
        # the TRIBE calls the caller already chose to make.
        pairs = same_track_excerpts if isinstance(same_track_excerpts, list) else [same_track_excerpts]
        floor_values = []
        for p1, p2 in pairs:
            d1 = _raw_delta_for_clip(model, p1, baseline_mean)
            d2 = _raw_delta_for_clip(model, p2, baseline_mean)
            prof1 = _profile_from_delta(d1, vertex_mask, network_labels)
            prof2 = _profile_from_delta(d2, vertex_mask, network_labels)
            floor_values.append(metric.distance(prof1, prof2, dynamics_mean, dynamics_std, weights).D_brain)
        floor = float(np.mean(floor_values))
        logger.info("Floor measured from %d same-track excerpt pair(s): mean=%.4f values=%s",
                    len(floor_values), floor, [f"{v:.4f}" for v in floor_values])
    else:
        # fall back to the smallest observed pairwise distance in the library
        # as a conservative (likely too-high) proxy — flagged clearly so it's
        # never mistaken for a real same-track measurement.
        floor = float(np.min(null_distribution))
        logger.warning(
            "No same-track excerpt pair provided — using min(null_distribution)=%.4f as a "
            "placeholder floor. This OVERSTATES the achievable floor; re-run build_calibration "
            "with same_track_excerpts for a real number.", floor,
        )

    return CalibrationBundle(
        baseline_mean=baseline_mean,
        vertex_mask=vertex_mask,
        network_labels=network_labels,
        dynamics_mean=dynamics_mean,
        dynamics_std=dynamics_std,
        null_distribution=null_distribution,
        floor=floor,
        anatomical_mask_size=int(anatomical_mask.sum()),
        data_driven_mask_size=int(data_driven_mask.sum()),
        overlap_size=overlap_size,
    )


def _profile_from_delta(delta, vertex_mask, network_labels) -> metric.BrainProfile:
    delta_masked = metric.apply_mask(delta, vertex_mask)
    roi_tc = metric._roi_mean_timecourse(delta_masked)
    network_labels_masked = network_labels[vertex_mask] if network_labels is not None else None
    return metric.BrainProfile(
        spatial=delta_masked.mean(axis=0),
        vertex_std=delta_masked.std(axis=0),
        spectrum=metric._low_freq_spectrum(roi_tc, metric.TIMESTEP_RATE_HZ),
        autocorr=metric._autocorr_lags(roi_tc),
        network_corr_upper=metric._network_correlation_upper(delta_masked, network_labels_masked),
        n_timesteps=delta_masked.shape[0],
    )


def score_against_reference(
    model,
    bundle: CalibrationBundle,
    ref_clip_path: Path,
    cand_clip_path: Path,
    weights: dict = metric.DEFAULT_WEIGHTS,
) -> tuple[metric.DistanceResult, metric.BrainProfile, metric.BrainProfile]:
    """Convenience wrapper for one-off comparisons (tests, ad-hoc scripts):
    scores one candidate against one reference, fully calibrated. Recomputes
    the reference's TRIBE pass every call — for the optimizer loop, where the
    same reference is scored against many candidates in a row, use
    `score_candidate_against_profile` with a `compute_profile_for_clip`
    result computed ONCE instead, or this doubles TRIBE cost per candidate
    for no reason (see FINDINGS.md §8).
    """
    ref_profile = compute_profile_for_clip(model, bundle, ref_clip_path)
    cand_profile = compute_profile_for_clip(model, bundle, cand_clip_path)
    result = metric.distance(ref_profile, cand_profile, bundle.dynamics_mean, bundle.dynamics_std, weights)
    result = metric.calibrate(result, bundle.null_distribution, bundle.floor)
    return result, ref_profile, cand_profile


def compute_profile_for_clip(model, bundle: CalibrationBundle, clip_path: Path) -> metric.BrainProfile:
    """One TRIBE forward pass -> a calibrated BrainProfile for one clip.
    Call once per clip and reuse the result — this is the expensive step."""
    delta = _raw_delta_for_clip(model, clip_path, bundle.baseline_mean)
    return _profile_from_delta(delta, bundle.vertex_mask, bundle.network_labels)


def score_candidate_against_profile(
    bundle: CalibrationBundle,
    ref_profile: metric.BrainProfile,
    cand_profile: metric.BrainProfile,
    weights: dict = metric.DEFAULT_WEIGHTS,
) -> metric.DistanceResult:
    """Pure comparison of two already-computed profiles — no TRIBE call.
    Use this in any loop that scores many candidates against one fixed
    reference (e.g. the optimizer), with the reference profile computed once
    via `compute_profile_for_clip`."""
    result = metric.distance(ref_profile, cand_profile, bundle.dynamics_mean, bundle.dynamics_std, weights)
    return metric.calibrate(result, bundle.null_distribution, bundle.floor)
