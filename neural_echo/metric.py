"""The brain-cost function. Pure functions only — no I/O, no TRIBE calls.

Compares a candidate's and a benchmark's raw TRIBE predictions region-by-region:
each of ~50 anatomical lobule groups (atlases.build_lobule_regions, left/right
hemispheres separate) is temporally summarized into N_TIME_WINDOWS equal time
bins, then compared as a (region, window) matrix — an L2-normalized distance
between the two arcs plus (1 - their correlation), averaged across regions
into a single global_score. Lower is better; there is no upper bound and no
external calibration (no baseline subtraction, no vertex masking, no null
distribution) — the score is self-normalizing per region via division by the
benchmark's own norm.

Everything below operates on `preds`: a (n_timesteps, 20484) array as returned
by TribeModel.predict().
"""
from dataclasses import dataclass, field

import numpy as np

N_TIME_WINDOWS = 12


def summarize_by_region_and_window(
    preds: np.ndarray, regions: dict[str, np.ndarray], n_windows: int = N_TIME_WINDOWS
) -> np.ndarray:
    """(n_timesteps, 20484) -> (n_regions, n_windows): mean over each region's
    vertices, then mean-pooled into n_windows equal time bins. Region order is
    sorted region-name order, consistent across calls."""
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim != 2:
        raise ValueError(f"Expected (n_timesteps, n_vertices), got shape {preds.shape}")
    if preds.shape[0] < n_windows:
        raise ValueError(f"Clip has {preds.shape[0]} timesteps, need at least n_windows={n_windows}")

    names = sorted(regions.keys())
    per_region_time = np.stack([preds[:, regions[n]].mean(axis=1) for n in names], axis=1)  # (T, n_regions)
    edges = np.linspace(0, preds.shape[0], n_windows + 1).astype(int)
    return np.stack(
        [per_region_time[edges[i]:edges[i + 1]].mean(axis=0) for i in range(n_windows)], axis=1,
    )  # (n_regions, n_windows)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


@dataclass
class RegionScore:
    region: str
    distance: float
    arc_correlation: float
    score: float


@dataclass
class WindowSummary:
    window_index: int
    rms_error: float
    mean_bias: float


@dataclass
class WorstCell:
    window_index: int
    region: str
    candidate: float
    target: float
    difference: float


@dataclass
class Cell:
    window_index: int
    region: str
    candidate: float
    target: float
    difference: float


@dataclass
class CostResult:
    global_score: float  # mean of all RegionScore.score — lower is better; what the optimizer minimizes
    regions: list[RegionScore]
    windows: list[WindowSummary]
    worst_cell: WorstCell
    cells: list[Cell]  # full region x window matrix, for the LLM's per-iteration diagnostics
    laterality: dict[str, float] = field(default_factory=dict)


def compute_cost(candidate_preds: np.ndarray, benchmark_preds: np.ndarray, regions: dict[str, np.ndarray]) -> CostResult:
    C = summarize_by_region_and_window(candidate_preds, regions)
    B = summarize_by_region_and_window(benchmark_preds, regions)
    names = sorted(regions.keys())
    n_windows = C.shape[1]

    cell_values = {
        (v, n): {
            "candidate": float(C[i, v]), "target": float(B[i, v]), "difference": float(C[i, v] - B[i, v]),
        }
        for i, n in enumerate(names) for v in range(n_windows)
    }

    region_scores: list[RegionScore] = []
    for i, n in enumerate(names):
        distance = float(np.linalg.norm(C[i] - B[i]) / (np.linalg.norm(B[i]) + 1e-8))
        arc_corr = _correlation(C[i], B[i])
        region_scores.append(RegionScore(region=n, distance=round(distance, 4), arc_correlation=round(arc_corr, 4),
                                          score=round(distance + (1 - arc_corr), 4)))

    windows: list[WindowSummary] = []
    for v in range(n_windows):
        diffs = np.array([cell_values[(v, n)]["difference"] for n in names])
        windows.append(WindowSummary(window_index=v, rms_error=float(np.sqrt(np.mean(diffs ** 2))),
                                      mean_bias=float(diffs.mean())))

    worst_key = max(cell_values, key=lambda k: abs(cell_values[k]["difference"]))
    worst_cell = WorstCell(window_index=worst_key[0], region=worst_key[1], **cell_values[worst_key])
    cells = [Cell(window_index=v, region=n, **cell_values[(v, n)]) for (v, n) in cell_values]

    score_by_region = {rs.region: rs.score for rs in region_scores}
    groups = {n.rsplit("_", 1)[0] for n in names}
    laterality = {}
    for g in groups:
        left, right = score_by_region.get(f"{g}_left"), score_by_region.get(f"{g}_right")
        if left is not None and right is not None:
            laterality[g] = round(left - right, 4)

    global_score = float(np.mean([rs.score for rs in region_scores]))
    return CostResult(global_score=global_score, regions=region_scores, windows=windows,
                       worst_cell=worst_cell, cells=cells, laterality=laterality)


def format_cost_for_llm(result: CostResult, iteration: int | None = None) -> str:
    lines = []
    if iteration:
        lines.append(f"=== Iteration {iteration} ===")
    lines.append(f"Global score: {result.global_score:.4f} (lower is better)")
    lines.append("cand=candidate, target=benchmark, diff=cand-target\n")

    region_names = sorted({c.region for c in result.cells})
    n_windows = len(result.windows)
    cells_by_window: dict[int, dict[str, Cell]] = {}
    for c in result.cells:
        cells_by_window.setdefault(c.window_index, {})[c.region] = c

    lines.append(f"MATRIX {n_windows} windows x {len(region_names)} regions:\n")
    for w in result.windows:
        pct_start, pct_end = int(100 * w.window_index / n_windows), int(100 * (w.window_index + 1) / n_windows)
        lines.append(
            f"[Window {w.window_index + 1:>2}/{n_windows} ({pct_start:>3}-{pct_end:>3}%)  "
            f"rms={w.rms_error:.3f}  bias={w.mean_bias:+.3f}]"
        )
        for region in region_names:
            c = cells_by_window[w.window_index][region]
            marker = "  <== worst" if (w.window_index == result.worst_cell.window_index and region == result.worst_cell.region) else ""
            lines.append(f"  {region:<24} cand={c.candidate:+.3f}  target={c.target:+.3f}  diff={c.difference:+.3f}{marker}")
        lines.append("")

    lines.append("Per-region summary:")
    for rs in result.regions:
        lines.append(f"  {rs.region:<24} dist={rs.distance:.3f}  arc_corr={rs.arc_correlation:+.3f}  score={rs.score:.3f}")

    if result.laterality:
        lines.append("\nLeft/right asymmetry (left_score - right_score; positive = left scored worse):")
        for group, asym in sorted(result.laterality.items()):
            lines.append(f"  {group:<24} {asym:+.4f}")

    return "\n".join(lines)
