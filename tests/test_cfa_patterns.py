from __future__ import annotations

import numpy as np

from mini_isp.io_utils import Frame
from mini_isp.stages import cfa_index_map, stage_demosaic, stage_wb_gains


def _make_mosaic(pattern: str, r: float, g: float, b: float, size: int = 4) -> np.ndarray:
    mosaic = np.zeros((size, size), dtype=np.float32)
    cfa = cfa_index_map(pattern)
    r_pos = cfa["R"]
    b_pos = cfa["B"]
    g1_pos = cfa["G1"]
    g2_pos = cfa["G2"]
    mosaic[r_pos[0]::2, r_pos[1]::2] = r
    mosaic[b_pos[0]::2, b_pos[1]::2] = b
    mosaic[g1_pos[0]::2, g1_pos[1]::2] = g
    mosaic[g2_pos[0]::2, g2_pos[1]::2] = g
    return mosaic


def test_wb_gains_patterns() -> None:
    patterns = ["RGGB", "BGGR", "GRBG", "GBRG"]
    for pattern in patterns:
        mosaic = _make_mosaic(pattern, r=1.0, g=2.0, b=3.0)
        frame = Frame(image=mosaic, meta={"cfa_pattern": pattern})
        result = stage_wb_gains(frame, {"wb_gains": [2.0, 3.0, 5.0]})
        out = result.frame.image
        cfa = cfa_index_map(pattern)
        assert np.all(out[cfa["R"][0]::2, cfa["R"][1]::2] == 2.0)
        assert np.all(out[cfa["B"][0]::2, cfa["B"][1]::2] == 15.0)
        assert np.all(out[cfa["G1"][0]::2, cfa["G1"][1]::2] == 6.0)
        assert np.all(out[cfa["G2"][0]::2, cfa["G2"][1]::2] == 6.0)


def test_demosaic_respects_pattern() -> None:
    patterns = ["RGGB", "BGGR", "GRBG", "GBRG"]
    for pattern in patterns:
        mosaic = _make_mosaic(pattern, r=0.1, g=0.2, b=0.3)
        frame = Frame(image=mosaic, meta={"cfa_pattern": pattern})
        rgb = stage_demosaic(frame, {"method": "bilinear"}).frame.image
        assert rgb.dtype == np.float32
        assert rgb.shape == (4, 4, 3)
        top_left = mosaic[0, 0]
        if pattern == "RGGB":
            assert np.isclose(rgb[0, 0, 0], top_left)
        elif pattern == "BGGR":
            assert np.isclose(rgb[0, 0, 2], top_left)
        elif pattern == "GRBG":
            assert np.isclose(rgb[0, 0, 1], top_left)
        elif pattern == "GBRG":
            assert np.isclose(rgb[0, 0, 1], top_left)

