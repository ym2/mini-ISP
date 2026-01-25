from __future__ import annotations

import numpy as np

from mini_isp.io_utils import Frame
from mini_isp.stages import stage_ccm, stage_stats_3a


def test_ccm_identity_no_change() -> None:
    img = np.random.default_rng(0).random((4, 4, 3), dtype=np.float32)
    frame = Frame(image=img, meta={})
    result = stage_ccm(frame, {"mode": "identity"})
    assert np.allclose(result.frame.image, img)
    assert result.frame.meta["ccm_mode"] == "identity"
    assert np.array(result.frame.meta["ccm"]).shape == (3, 3)


def test_ccm_manual_matrix() -> None:
    img = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    matrix = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    frame = Frame(image=img, meta={})
    result = stage_ccm(frame, {"mode": "manual", "matrix": matrix})
    out = result.frame.image
    assert np.allclose(out[0, 0], [0.0, 1.0, 0.0])


def test_stats_3a_non_invasive_and_keys() -> None:
    img = np.random.default_rng(1).random((8, 8, 3), dtype=np.float32)
    frame = Frame(image=img, meta={})
    result = stage_stats_3a(frame, {})
    assert np.array_equal(result.frame.image, img)

    stats = result.metrics["stats_3a"]
    assert "ae" in stats and "awb" in stats and "af" in stats
    assert len(stats["ae"]["hist"]) == 64
    assert np.isfinite(stats["af"]["tenengrad"])
