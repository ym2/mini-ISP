from __future__ import annotations

import numpy as np

from mini_isp.io_utils import Frame
from mini_isp.stages import stage_demosaic, stage_wb_gains


def test_wb_gains_rggb() -> None:
    mosaic = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    frame = Frame(image=mosaic, meta={"cfa_pattern": "RGGB"})
    result = stage_wb_gains(frame, {"wb_gains": [2.0, 3.0, 4.0]})
    out = result.frame.image

    expected = np.array([[2.0, 6.0], [9.0, 16.0]], dtype=np.float32)
    assert np.allclose(out, expected)
    assert result.frame.meta["wb_gains"] == [2.0, 3.0, 4.0]
    assert result.frame.meta["wb_applied"] is True
    assert "wb_gains" in result.metrics
    assert "wb_applied" in result.metrics


def _edge_clamp_neighbors(mosaic: np.ndarray, y: int, x: int) -> dict:
    h, w = mosaic.shape
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))
    y0 = clamp(y - 1, 0, h - 1)
    y1 = clamp(y + 1, 0, h - 1)
    x0 = clamp(x - 1, 0, w - 1)
    x1 = clamp(x + 1, 0, w - 1)
    return {
        "n": mosaic[y0, x],
        "s": mosaic[y1, x],
        "w": mosaic[y, x0],
        "e": mosaic[y, x1],
        "nw": mosaic[y0, x0],
        "ne": mosaic[y0, x1],
        "sw": mosaic[y1, x0],
        "se": mosaic[y1, x1],
    }


def test_demosaic_bilinear_shape_and_border() -> None:
    mosaic = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    frame = Frame(image=mosaic, meta={"cfa_pattern": "RGGB"})
    result = stage_demosaic(frame, {"method": "bilinear"})
    rgb = result.frame.image

    assert rgb.shape == (3, 3, 3)
    assert rgb.dtype == np.float32
    assert np.min(rgb) >= 0.0
    assert np.max(rgb) <= 9.0

    # Top-left corner is an R site in RGGB
    neighbors = _edge_clamp_neighbors(mosaic, 0, 0)
    expected_g = 0.25 * (neighbors["n"] + neighbors["s"] + neighbors["w"] + neighbors["e"])
    expected_b = 0.25 * (neighbors["nw"] + neighbors["ne"] + neighbors["sw"] + neighbors["se"])

    assert np.isclose(rgb[0, 0, 0], mosaic[0, 0])  # R channel matches
    assert np.isclose(rgb[0, 0, 1], expected_g)
    assert np.isclose(rgb[0, 0, 2], expected_b)


def test_demosaic_malvar_not_implemented() -> None:
    mosaic = np.ones((4, 4), dtype=np.float32)
    frame = Frame(image=mosaic, meta={"cfa_pattern": "RGGB"})
    try:
        stage_demosaic(frame, {"method": "malvar"})
    except NotImplementedError as exc:
        assert "malvar" in str(exc).lower()
    else:
        raise AssertionError("Expected NotImplementedError for malvar method")
