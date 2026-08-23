"""Plotly brain-response figures for optimizer progress.

The full TRIBE tensor is intentionally not kept after scoring.  Instead, each
iteration stores one signed residual value per fsaverage5 vertex: temporal RMS
error supplies the magnitude and the prevailing / strongest error supplies the
sign.  That is small enough to retain in memory and is exactly the quantity the
UI needs to tell the convergence story spatially.
"""
# Plotly's own schema examples conventionally use dict(...) for deeply nested
# layout objects; it stays substantially more readable here than quoted keys.
# ruff: noqa: C408
from __future__ import annotations

from functools import lru_cache

import numpy as np

N_VERTICES = 20_484
PAPER_BG = "#090b12"
POSITIVE_RGB = np.array([255, 91, 104], dtype=float)   # candidate above reference
NEGATIVE_RGB = np.array([65, 207, 255], dtype=float)  # candidate below reference
MID_RGB = np.array([222, 228, 240], dtype=float)


def summarize_vertex_residual(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Collapse two ``(time, vertex)`` TRIBE predictions into a signed map.

    Clips can occasionally differ by a timestep after feature extraction, so
    both are sampled at equal relative positions before subtraction.  RMS keeps
    oscillating mismatches visible; mean bias provides the sign when stable,
    otherwise the strongest instantaneous mismatch does.
    """
    candidate = np.asarray(candidate, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if candidate.ndim != 2 or reference.ndim != 2:
        raise ValueError("Brain predictions must have shape (time, vertices)")
    if candidate.shape[1] != reference.shape[1]:
        raise ValueError("Candidate and reference must use the same cortical mesh")

    n_time = min(candidate.shape[0], reference.shape[0])
    if n_time == 0:
        raise ValueError("Brain predictions cannot be empty")
    candidate_idx = np.rint(np.linspace(0, candidate.shape[0] - 1, n_time)).astype(int)
    reference_idx = np.rint(np.linspace(0, reference.shape[0] - 1, n_time)).astype(int)
    delta = candidate[candidate_idx] - reference[reference_idx]

    rms = np.sqrt(np.mean(np.square(delta, dtype=np.float32), axis=0))
    bias = np.mean(delta, axis=0)
    strongest_idx = np.argmax(np.abs(delta), axis=0)
    strongest = delta[strongest_idx, np.arange(delta.shape[1])]
    sign_source = np.where(np.abs(bias) >= 0.05 * rms, bias, strongest)
    return (rms * np.sign(sign_source)).astype(np.float32)


def reaction_vertexcolors(
    values: np.ndarray,
    *,
    threshold: float,
    soft_edge: float,
    vmax: float,
) -> list[str]:
    """Create per-vertex RGBA with genuine transparency below threshold."""
    values = np.asarray(values, dtype=float)
    magnitude = np.abs(values)
    normalized = np.clip(magnitude / max(vmax, 1e-8), 0.0, 1.0)
    # A pale inner glow becomes saturated at the hottest residuals.  The alpha
    # ramp is independent of color, preserving the notebook's key visual trick.
    target = np.where(values[:, None] >= 0, POSITIVE_RGB, NEGATIVE_RGB)
    rgb = MID_RGB[None, :] + (target - MID_RGB[None, :]) * np.sqrt(normalized)[:, None]
    alpha = np.clip((magnitude - threshold) / max(soft_edge, 1e-8), 0.0, 1.0)
    rgb = np.rint(rgb).astype(int)
    return [f"rgba({r},{g},{b},{a:.3f})" for (r, g, b), a in zip(rgb, alpha)]


@lru_cache(maxsize=1)
def _base_figure():
    """Build and cache the static inflated fsaverage anatomy trace."""
    from nilearn import datasets, plotting
    from nilearn.surface import PolyData, PolyMesh, SurfaceImage

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    brain_mesh = PolyMesh(left=fsaverage.infl_left, right=fsaverage.infl_right)
    background = SurfaceImage(
        mesh=brain_mesh,
        data=PolyData(left=fsaverage.sulc_left, right=fsaverage.sulc_right),
    )
    view = plotting.plot_surf(
        surf_mesh=brain_mesh,
        surf_map=None,
        bg_map=background,
        hemi="both",
        engine="plotly",
        colorbar=False,
    )
    trace = view.figure.data[0]
    trace.opacity = 0.22
    trace.flatshading = False
    trace.lighting = dict(ambient=0.52, diffuse=0.78, specular=0.72, roughness=0.42, fresnel=0.18)
    trace.lightposition = dict(x=140, y=190, z=160)
    trace.hoverinfo = "skip"
    return view.figure


def build_brain_response_figure(
    residuals: list[np.ndarray], iteration_indices: list[int]
) -> tuple[object, dict]:
    """Return an animated Plotly figure plus lightweight convergence metadata."""
    import plotly.graph_objects as go

    if not residuals or len(residuals) != len(iteration_indices):
        raise ValueError("A residual map and iteration index are required for every frame")
    maps = [np.asarray(values, dtype=np.float32) for values in residuals]
    if any(values.shape != (N_VERTICES,) for values in maps):
        raise ValueError(f"Every residual map must have {N_VERTICES} vertices")

    all_magnitudes = np.abs(np.concatenate(maps))
    vmax = max(float(np.percentile(all_magnitudes, 99.3)), 1e-6)
    threshold = max(float(np.percentile(all_magnitudes, 58)), vmax * 0.12)
    soft_edge = max(vmax * 0.24, 1e-6)
    colors = [
        reaction_vertexcolors(values, threshold=threshold, soft_edge=soft_edge, vmax=vmax)
        for values in maps
    ]

    # Copy so concurrent jobs can safely customize layout/frames while sharing
    # the expensive-to-construct fsaverage geometry.
    fig = go.Figure(_base_figure())
    base = fig.data[0]
    fig.add_trace(go.Mesh3d(
        x=base.x, y=base.y, z=base.z,
        i=base.i, j=base.j, k=base.k,
        vertexcolor=colors[-1],
        opacity=1,
        flatshading=False,
        lighting=dict(ambient=0.48, diffuse=0.9, specular=1, roughness=0.28, fresnel=0.3),
        lightposition=dict(x=140, y=190, z=160),
        hoverinfo="skip",
        name="Residual mismatch",
    ))

    frame_names = [str(index) for index in iteration_indices]
    fig.frames = [
        go.Frame(
            name=name,
            traces=[1],
            data=[go.Mesh3d(vertexcolor=frame_colors)],
        )
        for name, frame_colors in zip(frame_names, colors)
    ]

    def play_button(speed: float) -> dict:
        return dict(
            label=f"▶  {speed:g}×",
            method="animate",
            args=[None, dict(
                frame=dict(duration=int(1050 / speed), redraw=True),
                transition=dict(duration=140),
                # Replay the convergence story from the beginning even though
                # the live view intentionally opens on the latest frame.
                fromcurrent=False,
                mode="immediate",
            )],
        )

    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        showlegend=False,
        uirevision="brain-camera",
        scene=dict(
            bgcolor=PAPER_BG,
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.38, y=1.32, z=0.82), up=dict(x=0, y=0, z=1)),
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, t=4, b=86),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.02, y=0.055,
            xanchor="left", yanchor="middle",
            pad=dict(r=8, t=4, b=4, l=8),
            bgcolor="rgba(21,25,38,0.92)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(color="#eef2ff", size=11),
            buttons=[
                dict(
                    label="Ⅱ",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        transition=dict(duration=0),
                        mode="immediate",
                    )],
                ),
                play_button(1),
                play_button(2),
            ],
        )],
        sliders=[dict(
            active=len(frame_names) - 1,
            x=0.35, y=0.055, len=0.62,
            xanchor="left", yanchor="middle",
            pad=dict(t=8, b=0),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            tickcolor="rgba(255,255,255,0.14)",
            font=dict(color="#8d93a6", size=10),
            currentvalue=dict(
                prefix="ITERATION  ",
                visible=True,
                xanchor="right",
                font=dict(color="#f8fafc", size=11),
            ),
            steps=[dict(
                method="animate",
                label=name.zfill(2),
                args=[[name], dict(
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=100),
                    mode="immediate",
                )],
            ) for name in frame_names],
        )],
    )

    summaries = []
    for index, values in zip(iteration_indices, maps):
        magnitude = np.abs(values)
        summaries.append({
            "iteration_index": index,
            "mean_mismatch": float(np.mean(magnitude)),
            "peak_mismatch": float(np.max(magnitude)),
            "active_fraction": float(np.mean(magnitude > threshold)),
        })
    return fig, {
        "frames": summaries,
        "threshold": threshold,
        "scale_max": vmax,
        "latest_iteration": iteration_indices[-1],
    }
