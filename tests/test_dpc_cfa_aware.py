from __future__ import annotations

import numpy as np
import pytest

from mini_isp.io_utils import Frame
from mini_isp.stages import cfa_index_map, inject_defects_for_test, stage_dpc


def _make_cfa_plane_mosaic(
    pattern: str, r: float, g1: float, g2: float, b: float, size: int = 10
) -> np.ndarray:
    mosaic = np.zeros((size, size), dtype=np.float32)
    cfa = cfa_index_map(pattern)
    mosaic[cfa["R"][0]::2, cfa["R"][1]::2] = r
    mosaic[cfa["G1"][0]::2, cfa["G1"][1]::2] = g1
    mosaic[cfa["G2"][0]::2, cfa["G2"][1]::2] = g2
    mosaic[cfa["B"][0]::2, cfa["B"][1]::2] = b
    return mosaic


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_dpc_same_cfa_only_no_false_fixes(pattern: str) -> None:
    mosaic = _make_cfa_plane_mosaic(pattern, r=0.1, g1=0.4, g2=0.6, b=0.9)
    frame = Frame(image=mosaic, meta={"cfa_pattern": pattern})
    result = stage_dpc(frame, {"threshold": 0.2})
    out = result.frame.image

    assert result.metrics["neighbor_policy"] == "same_cfa_only"
    assert result.metrics["neighbor_stat"] == "median"
    assert result.metrics["n_fixed"] == 0
    assert np.allclose(out, mosaic)


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize("plane", ["R", "G1", "G2", "B"])
def test_dpc_same_cfa_only_repairs_single_defect(pattern: str, plane: str) -> None:
    base = _make_cfa_plane_mosaic(pattern, r=0.1, g1=0.4, g2=0.6, b=0.9)
    cfa = cfa_index_map(pattern)
    coord = (cfa[plane][0] + 4, cfa[plane][1] + 4)
    base_value = float(base[coord])
    defect_value = 1.0 if base_value < 0.8 else 0.0

    defected = inject_defects_for_test(base, (coord,), (defect_value,))
    frame = Frame(image=defected, meta={"cfa_pattern": pattern})
    result = stage_dpc(frame, {"threshold": 0.2})
    out = result.frame.image

    assert result.metrics["n_fixed"] == 1
    assert np.isclose(out[coord], base_value, atol=0.0)
    assert np.allclose(out, base)

