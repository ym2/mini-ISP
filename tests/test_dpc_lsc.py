from __future__ import annotations

import numpy as np

from mini_isp.io_utils import Frame
from mini_isp.stages import inject_defects_for_test, stage_dpc, stage_lsc


def test_dpc_fixes_injected_defects() -> None:
    base = np.full((8, 8), 0.5, dtype=np.float32)
    coords = ((1, 1), (2, 3), (6, 6))
    values = (0.0, 1.0, 0.0)
    defected = inject_defects_for_test(base, coords, values)
    frame = Frame(image=defected, meta={})
    result = stage_dpc(frame, {"threshold": 0.2})
    corrected = result.frame.image

    assert result.metrics["n_fixed"] == len(coords)
    assert corrected.dtype == np.float32
    assert np.allclose(corrected, base)


def test_lsc_gain_cap_and_symmetry() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    frame = Frame(image=image, meta={})
    result = stage_lsc(frame, {"gain_cap": 1.5, "k": 0.8})
    out = result.frame.image

    assert out.dtype == np.float32
    assert float(np.max(out)) <= 1.5 + 1e-6
    assert np.isclose(out[4, 4], 1.0, atol=1e-6)
    assert np.isclose(out[0, 0], out[0, -1], atol=1e-6)
    assert np.isclose(out[0, 0], out[-1, 0], atol=1e-6)
    assert np.isclose(out[0, 1], out[1, 0], atol=1e-6)
