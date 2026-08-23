"""Premium cortical comparison figures for optimizer progress."""

# Plotly layout is materially clearer with dict(...) than deeply quoted keys.
# ruff: noqa: C408
from __future__ import annotations

from functools import lru_cache

import numpy as np

N_VERTICES = 20_484
PAPER_BG = "#090b12"
POSITIVE_RGB = np.array([255, 91, 104], dtype=float)
NEGATIVE_RGB = np.array([65, 207, 255], dtype=float)
MID_RGB = np.array([222, 228, 240], dtype=float)


def _align_predictions(candidate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    return candidate[candidate_idx], reference[reference_idx]


def summarize_vertex_residual(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Collapse two TRIBE predictions into a signed temporal mismatch map."""
    candidate_aligned, reference_aligned = _align_predictions(candidate, reference)
    delta = candidate_aligned - reference_aligned
    rms = np.sqrt(np.mean(np.square(delta, dtype=np.float32), axis=0))
    bias = np.mean(delta, axis=0)
    strongest_idx = np.argmax(np.abs(delta), axis=0)
    strongest = delta[strongest_idx, np.arange(delta.shape[1])]
    sign_source = np.where(np.abs(bias) >= 0.05 * rms, bias, strongest)
    return (rms * np.sign(sign_source)).astype(np.float32)


def summarize_vertex_activity(prediction: np.ndarray) -> np.ndarray:
    """Collapse a TRIBE prediction into temporal RMS activity per vertex."""
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim != 2 or prediction.shape[0] == 0:
        raise ValueError("Brain predictions must have non-empty shape (time, vertices)")
    return np.sqrt(np.mean(np.square(prediction, dtype=np.float32), axis=0)).astype(np.float32)


def reaction_vertexcolors(
    values: np.ndarray, *, threshold: float, soft_edge: float, vmax: float
) -> list[str]:
    """Create diverging per-vertex RGBA with genuine transparency."""
    values = np.asarray(values, dtype=float)
    magnitude = np.abs(values)
    normalized = np.clip(magnitude / max(vmax, 1e-8), 0.0, 1.0)
    target = np.where(values[:, None] >= 0, POSITIVE_RGB, NEGATIVE_RGB)
    rgb = MID_RGB[None, :] + (target - MID_RGB[None, :]) * np.sqrt(normalized)[:, None]
    alpha = np.clip((magnitude - threshold) / max(soft_edge, 1e-8), 0.0, 1.0)
    rgb = np.rint(rgb).astype(int)
    return [f"rgba({r},{g},{b},{a:.3f})" for (r, g, b), a in zip(rgb, alpha)]


def activity_vertexcolors(
    values: np.ndarray, *, threshold: float, soft_edge: float, vmax: float
) -> list[str]:
    """Inferno activity colors with transparent inactive cortex."""
    from matplotlib import colormaps

    values = np.asarray(values, dtype=float)
    normalized = np.clip(values / max(vmax, 1e-8), 0.0, 1.0)
    rgba = colormaps["inferno"](normalized)
    alpha = np.clip((values - threshold) / max(soft_edge, 1e-8), 0.0, 0.96)
    rgb = np.rint(rgba[:, :3] * 255).astype(int)
    return [f"rgba({r},{g},{b},{a:.3f})" for (r, g, b), a in zip(rgb, alpha)]


@lru_cache(maxsize=1)
def _base_trace():
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
    return view.figure.data[0]


def build_brain_response_figure(
    reference_activity: np.ndarray,
    candidate_activities: list[np.ndarray],
    residuals: list[np.ndarray],
    iteration_indices: list[int],
) -> tuple[object, dict]:
    """Build an animated shared-scale reference/candidate brain comparison."""
    import plotly.graph_objects as go

    if (
        not residuals
        or len(residuals) != len(iteration_indices)
        or len(candidate_activities) != len(iteration_indices)
    ):
        raise ValueError("Activity, residual, and iteration index are required for every frame")
    reference_map = np.asarray(reference_activity, dtype=np.float32)
    activity_maps = [np.asarray(values, dtype=np.float32) for values in candidate_activities]
    residual_maps = [np.asarray(values, dtype=np.float32) for values in residuals]
    if reference_map.shape != (N_VERTICES,) or any(
        values.shape != (N_VERTICES,) for values in [*activity_maps, *residual_maps]
    ):
        raise ValueError(f"Every cortical map must have {N_VERTICES} vertices")

    residual_magnitudes = np.abs(np.concatenate(residual_maps))
    residual_vmax = max(float(np.percentile(residual_magnitudes, 99.3)), 1e-6)
    residual_threshold = max(
        float(np.percentile(residual_magnitudes, 58)), residual_vmax * 0.12
    )

    all_activity = np.concatenate([reference_map, *activity_maps])
    activity_vmax = max(float(np.percentile(all_activity, 99.3)), 1e-6)
    activity_threshold = max(float(np.percentile(all_activity, 42)), activity_vmax * 0.08)
    activity_soft_edge = max(activity_vmax * 0.2, 1e-6)
    reference_colors = activity_vertexcolors(
        reference_map,
        threshold=activity_threshold,
        soft_edge=activity_soft_edge,
        vmax=activity_vmax,
    )
    candidate_colors = [
        activity_vertexcolors(
            values,
            threshold=activity_threshold,
            soft_edge=activity_soft_edge,
            vmax=activity_vmax,
        )
        for values in activity_maps
    ]

    base = _base_trace()

    def anatomy_trace(scene: str):
        return go.Mesh3d(
            x=base.x, y=base.y, z=base.z,
            i=base.i, j=base.j, k=base.k,
            intensity=base.intensity,
            colorscale=base.colorscale,
            opacity=0.2,
            flatshading=False,
            lighting=dict(ambient=0.52, diffuse=0.78, specular=0.72, roughness=0.42, fresnel=0.18),
            lightposition=dict(x=140, y=190, z=160),
            hoverinfo="skip",
            scene=scene,
            showscale=False,
        )

    def activity_trace(colors: list[str], scene: str, name: str):
        return go.Mesh3d(
            x=base.x, y=base.y, z=base.z,
            i=base.i, j=base.j, k=base.k,
            vertexcolor=colors,
            opacity=1,
            flatshading=False,
            lighting=dict(ambient=0.48, diffuse=0.9, specular=1, roughness=0.28, fresnel=0.3),
            lightposition=dict(x=140, y=190, z=160),
            hoverinfo="skip",
            scene=scene,
            name=name,
            showscale=False,
        )

    fig = go.Figure()
    fig.add_trace(anatomy_trace("scene"))
    fig.add_trace(activity_trace(reference_colors, "scene", "Reference response"))
    fig.add_trace(anatomy_trace("scene2"))
    fig.add_trace(activity_trace(candidate_colors[-1], "scene2", "Candidate response"))

    frame_names = [str(index) for index in iteration_indices]
    fig.frames = [
        go.Frame(
            name=name,
            traces=[3],
            data=[go.Mesh3d(vertexcolor=colors, scene="scene2")],
        )
        for name, colors in zip(frame_names, candidate_colors)
    ]

    scene_style = dict(
        bgcolor=PAPER_BG,
        aspectmode="data",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        camera=dict(eye=dict(x=1.38, y=1.32, z=0.82), up=dict(x=0, y=0, z=1)),
        dragmode="orbit",
    )

    def play_button(speed: float) -> dict:
        return dict(
            label=f"▶  {speed:g}×",
            method="animate",
            args=[None, dict(
                frame=dict(duration=int(1050 / speed), redraw=True),
                transition=dict(duration=140),
                fromcurrent=False,
                mode="immediate",
            )],
        )

    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        showlegend=False,
        uirevision="brain-comparison-camera",
        scene={**scene_style, "domain": {"x": [0, 0.49], "y": [0.13, 0.94]}},
        scene2={**scene_style, "domain": {"x": [0.51, 1], "y": [0.13, 0.94]}},
        annotations=[
            dict(x=0.245, y=0.98, xref="paper", yref="paper", text="REFERENCE MEMORY",
                 showarrow=False, font=dict(color="#aeb4c5", size=11)),
            dict(x=0.755, y=0.98, xref="paper", yref="paper", text="EVOLVING CANDIDATE",
                 showarrow=False, font=dict(color="#ffffff", size=11)),
        ],
        margin=dict(l=0, r=0, t=8, b=86),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.02, y=0.055,
            xanchor="left", yanchor="middle", pad=dict(r=8, t=4, b=4, l=8),
            bgcolor="rgba(21,25,38,0.92)", bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1, font=dict(color="#eef2ff", size=11),
            buttons=[
                dict(label="Ⅱ", method="animate", args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    transition=dict(duration=0), mode="immediate",
                )]),
                play_button(1),
                play_button(2),
            ],
        )],
        sliders=[dict(
            active=len(frame_names) - 1, x=0.35, y=0.055, len=0.62,
            xanchor="left", yanchor="middle", pad=dict(t=8, b=0),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            tickcolor="rgba(255,255,255,0.14)", font=dict(color="#8d93a6", size=10),
            currentvalue=dict(prefix="ITERATION  ", visible=True, xanchor="right",
                              font=dict(color="#f8fafc", size=11)),
            steps=[dict(method="animate", label=name.zfill(2), args=[[name], dict(
                frame=dict(duration=0, redraw=True),
                transition=dict(duration=100), mode="immediate",
            )]) for name in frame_names],
        )],
    )

    summaries = []
    for index, values in zip(iteration_indices, residual_maps):
        magnitude = np.abs(values)
        summaries.append({
            "iteration_index": index,
            "mean_mismatch": float(np.mean(magnitude)),
            "peak_mismatch": float(np.max(magnitude)),
            "active_fraction": float(np.mean(magnitude > residual_threshold)),
        })
    return fig, {
        "frames": summaries,
        "threshold": residual_threshold,
        "scale_max": residual_vmax,
        "activity_scale_max": activity_vmax,
        "latest_iteration": iteration_indices[-1],
    }
