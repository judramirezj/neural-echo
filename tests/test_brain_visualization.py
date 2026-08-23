import numpy as np

from neural_echo.brain_visualization import (
    reaction_vertexcolors,
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
