import numpy as np
import plotly.graph_objects as go

from neural_echo import brain_visualization
from neural_echo.brain_visualization import (
    activity_vertexcolors,
    build_brain_response_figure,
    reaction_vertexcolors,
    summarize_vertex_activity,
    summarize_vertex_residual,
)


def test_vertex_residual_preserves_temporal_error_and_direction():
    reference = np.zeros((3, 4), dtype=np.float32)
    candidate = np.array([
        [1.0, -2.0, 1.0, 0.0],
        [1.0, -2.0, -1.0, 0.0],
        [1.0, -2.0, 1.0, 0.0],
    ], dtype=np.float32)

    residual = summarize_vertex_residual(candidate, reference)

    np.testing.assert_allclose(residual[:2], [1.0, -2.0])
    assert residual[2] > 0  # oscillating mismatch remains visible
    assert residual[3] == 0


def test_vertex_colors_have_real_soft_edge_transparency():
    colors = reaction_vertexcolors(
        np.array([0.0, 0.2, 0.4, -0.4]),
        threshold=0.2,
        soft_edge=0.2,
        vmax=0.4,
    )

    assert colors[0].endswith(",0.000)")
    assert colors[1].endswith(",0.000)")
    assert colors[2].endswith(",1.000)")
    assert colors[3].endswith(",1.000)")
    assert colors[2] != colors[3]  # positive coral vs negative cyan


def test_activity_summary_and_inferno_transparency():
    prediction = np.array([[0.0, 3.0], [0.0, 4.0]], dtype=np.float32)
    activity = summarize_vertex_activity(prediction)
    np.testing.assert_allclose(activity, [0.0, np.sqrt(12.5)])

    colors = activity_vertexcolors(activity, threshold=0.2, soft_edge=0.4, vmax=4.0)
    assert colors[0].endswith(",0.000)")
    assert not colors[1].endswith(",0.000)")


def test_comparison_figure_has_two_brains_and_animates_candidate(monkeypatch):
    monkeypatch.setattr(brain_visualization, "N_VERTICES", 4)
    monkeypatch.setattr(
        brain_visualization,
        "_base_trace",
        lambda: go.Mesh3d(
            x=[0, 1, 0, 1], y=[0, 0, 1, 1], z=[0, 0, 0, 0],
            i=[0, 1], j=[1, 2], k=[2, 3], intensity=[0, 1, 0, 1],
        ),
    )
    reference = np.array([0.1, 0.4, 0.8, 1.0], dtype=np.float32)
    candidates = [reference * 0.5, reference * 0.9]
    residuals = [reference * -0.5, reference * -0.1]

    figure, meta = build_brain_response_figure(reference, candidates, residuals, [1, 2])

    assert len(figure.data) == 4
    assert figure.data[1].scene == "scene"
    assert figure.data[3].scene == "scene2"
    assert len(figure.frames) == 2
    assert tuple(figure.frames[0].traces) == (3,)
    assert meta["latest_iteration"] == 2
