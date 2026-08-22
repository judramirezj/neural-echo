"""The brain-distance metric. Pure functions only — no I/O, no TRIBE calls.

Why a naive np.linalg.norm(preds_ref - preds_cand) is wrong, and what each step
here does about it (see project brief §3 for the full rationale):

  1. Length mismatch          -> length-invariant descriptors (spatial mean,
                                  temporal stats, network correlation), never
                                  raw per-timestep alignment.
  2. Global-mean domination   -> subtract a pink-noise/silence baseline before
                                  anything else (subtract_baseline).
  3. Vertex heteroscedasticity -> z-score dynamics descriptors against
                                  clip-library stats before combining them.
  4. Dead vertices             -> vertex_mask restricts everything to
                                  audio-relevant cortex (see atlases.py).
  5. Degenerate optimum        -> deliberately NOT handled here; novelty/
                                  adherence are separate hard filters applied
                                  downstream in the optimizer, never folded
                                  into D_brain (see generator.py / optimizer.py).

Everything below operates on `preds`: a (n_timesteps, 20484) array as returned
by TribeModel.predict(), at TIMESTEP_RATE_HZ (empirically 1.0, see FINDINGS.md).
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

TIMESTEP_RATE_HZ = 1.0  # confirmed empirically in Phase 0 — do not assume, re-verify if TRIBE changes
EDGE_TRIM_S = 6  # discard hemodynamic-lag edge artifacts (brief §3 step 0)
SPECTRUM_BAND_HZ = (0.01, 0.15)  # BOLD-relevant band; faster is measurement noise
N_SPECTRUM_BINS = 8
N_AUTOCORR_LAGS = 5

DEFAULT_WEIGHTS = dict(spatial=0.5, dynamics=0.2, geometry=0.3)


def trim_edges(preds: np.ndarray, rate_hz: float = TIMESTEP_RATE_HZ) -> np.ndarray:
    n_trim = int(round(EDGE_TRIM_S * rate_hz))
    if preds.shape[0] <= 2 * n_trim:
        return preds  # too short to trim; caller should use a longer clip
    return preds[n_trim: preds.shape[0] - n_trim]


def subtract_baseline(preds: np.ndarray, baseline_mean: np.ndarray) -> np.ndarray:
    """ΔX = X - baseline_mean, per brief §3 step 0b. baseline_mean is the
    per-vertex mean prediction over pink-noise + silence stimuli (see
    calibration.compute_baseline) — removes the "any audio" constant prior.
    """
    return preds - baseline_mean[None, :]


def apply_mask(delta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return delta[:, mask]


@dataclass
class BrainProfile:
    """Length-invariant descriptors of one clip's masked, baseline-subtracted
    brain response. Comparable across clips of different duration."""
    spatial: np.ndarray            # (k,) mean over time per vertex
    vertex_std: np.ndarray         # (k,) std over time per vertex
    spectrum: np.ndarray           # (N_SPECTRUM_BINS,) low-freq power of ROI-mean timecourse
    autocorr: np.ndarray           # (N_AUTOCORR_LAGS,) lag-1..lag-5 autocorr of ROI-mean timecourse
    network_corr_upper: np.ndarray  # (n_pairs,) vectorized upper triangle of network x network corr
    n_timesteps: int

    @property
    def dynamics_raw(self) -> np.ndarray:
        return np.concatenate([self.vertex_std, self.spectrum, self.autocorr])


def _roi_mean_timecourse(delta_masked: np.ndarray) -> np.ndarray:
    return delta_masked.mean(axis=1)


def _low_freq_spectrum(timecourse: np.ndarray, rate_hz: float) -> np.ndarray:
    n = len(timecourse)
    if n < 4:
        return np.zeros(N_SPECTRUM_BINS)
    windowed = timecourse - timecourse.mean()
    windowed = windowed * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / rate_hz)

    lo, hi = SPECTRUM_BAND_HZ
    edges = np.linspace(lo, hi, N_SPECTRUM_BINS + 1)
    binned = np.zeros(N_SPECTRUM_BINS)
    for i in range(N_SPECTRUM_BINS):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if sel.any():
            binned[i] = spectrum[sel].mean()
        # else leave at 0 — frequency resolution (1/duration) can exceed bin
        # width for short clips; an empty bin is a real (if coarse) result,
        # not a bug.
    return binned


def _autocorr_lags(timecourse: np.ndarray, n_lags: int = N_AUTOCORR_LAGS) -> np.ndarray:
    n = len(timecourse)
    x = timecourse - timecourse.mean()
    denom = np.sum(x ** 2) + 1e-12
    out = np.zeros(n_lags)
    for lag in range(1, n_lags + 1):
        if lag >= n:
            break
        out[lag - 1] = np.sum(x[:-lag] * x[lag:]) / denom
    return out


def _network_correlation_upper(
    delta_masked: np.ndarray, network_labels_masked: np.ndarray | None
) -> np.ndarray:
    """Collapse masked vertices into network-mean timecourses, return the
    vectorized upper triangle of their T-length correlation matrix (brief §3
    step 2c — "representational geometry" / RSA-style comparison).

    If network_labels_masked is None (Yeo atlas unavailable), falls back to a
    single-network (whole-mask) timecourse, yielding a degenerate 1x1
    "network geometry" — d_geometry becomes uninformative (spearman on a
    single scalar), so callers should down-weight geometry when this happens.
    """
    if network_labels_masked is None:
        return np.array([1.0])  # degenerate but stable placeholder

    unique_networks = sorted(n for n in np.unique(network_labels_masked) if n > 0)
    if len(unique_networks) < 2:
        return np.array([1.0])

    network_timecourses = np.stack([
        delta_masked[:, network_labels_masked == net].mean(axis=1)
        for net in unique_networks
    ])  # (n_networks, T)

    if network_timecourses.shape[1] < 3:
        n = len(unique_networks)
        return np.zeros(n * (n - 1) // 2)

    corr = np.corrcoef(network_timecourses)
    iu = np.triu_indices_from(corr, k=1)
    return corr[iu]


def compute_profile(
    preds: np.ndarray,
    baseline_mean: np.ndarray,
    vertex_mask: np.ndarray,
    network_labels: np.ndarray | None,
    rate_hz: float = TIMESTEP_RATE_HZ,
) -> BrainProfile:
    trimmed = trim_edges(preds, rate_hz)
    delta = subtract_baseline(trimmed, baseline_mean)
    delta_masked = apply_mask(delta, vertex_mask)

    roi_timecourse = _roi_mean_timecourse(delta_masked)
    network_labels_masked = network_labels[vertex_mask] if network_labels is not None else None

    return BrainProfile(
        spatial=delta_masked.mean(axis=0),
        vertex_std=delta_masked.std(axis=0),
        spectrum=_low_freq_spectrum(roi_timecourse, rate_hz),
        autocorr=_autocorr_lags(roi_timecourse),
        network_corr_upper=_network_correlation_upper(delta_masked, network_labels_masked),
        n_timesteps=delta_masked.shape[0],
    )


@dataclass
class DistanceResult:
    D_brain: float
    d_spatial: float
    d_dynamics: float
    d_geometry: float
    percentile: float | None = None
    null_median: float | None = None
    floor: float | None = None


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(sp_stats.pearsonr(a, b)[0])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(sp_stats.spearmanr(a, b)[0])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def distance(
    ref: BrainProfile,
    cand: BrainProfile,
    dynamics_mean: np.ndarray,
    dynamics_std: np.ndarray,
    weights: dict = DEFAULT_WEIGHTS,
) -> DistanceResult:
    """Core comparison (brief §3 step 3). dynamics_mean/std come from the
    calibration clip library (calibration.py) — z-scoring dynamics against
    the library, not against this single pair, is what makes d_dynamics
    comparable across candidates.
    """
    d_spatial = 1.0 - _safe_pearson(ref.spatial, cand.spatial)

    std_safe = np.where(dynamics_std < 1e-9, 1.0, dynamics_std)
    z_ref = (ref.dynamics_raw - dynamics_mean) / std_safe
    z_cand = (cand.dynamics_raw - dynamics_mean) / std_safe
    d_dynamics = 1.0 - _cosine(z_ref, z_cand)

    if ref.network_corr_upper.shape != cand.network_corr_upper.shape:
        d_geometry = 1.0  # incomparable shapes -> maximally distant, not a crash
    else:
        d_geometry = 1.0 - _safe_spearman(ref.network_corr_upper, cand.network_corr_upper)

    D_brain = (
        weights["spatial"] * d_spatial
        + weights["dynamics"] * d_dynamics
        + weights["geometry"] * d_geometry
    )
    return DistanceResult(
        D_brain=D_brain, d_spatial=d_spatial, d_dynamics=d_dynamics, d_geometry=d_geometry
    )


def calibrate(result: DistanceResult, null_distribution: np.ndarray, floor: float) -> DistanceResult:
    percentile = float((null_distribution < result.D_brain).mean() * 100.0)
    result.percentile = percentile
    result.null_median = float(np.median(null_distribution))
    result.floor = floor
    return result


def per_network_deltas(
    profile: BrainProfile,
    ref_profile: BrainProfile,
    network_labels_masked: np.ndarray | None,
) -> dict:
    """Per-network z-scored engagement delta vs. reference, for the
    optimizer's LLM diagnostics payload (brief §4: "candidate under-engages
    auditory association by 0.4σ ..."). Returns {network_name: sigma_delta}.
    """
    from . import atlases

    if network_labels_masked is None:
        return {}
    out = {}
    for net in sorted(n for n in np.unique(network_labels_masked) if n > 0):
        sel = network_labels_masked == net
        if not sel.any():
            continue
        cand_mean = profile.spatial[sel].mean()
        ref_mean = ref_profile.spatial[sel].mean()
        ref_std = ref_profile.spatial[sel].std() + 1e-9
        out[atlases.network_name(int(net))] = float((cand_mean - ref_mean) / ref_std)
    return out
