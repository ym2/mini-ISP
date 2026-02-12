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


def test_ccm_chain_equivalence_to_manual_effective() -> None:
    img = np.random.default_rng(2).random((4, 4, 3), dtype=np.float32)
    frame = Frame(image=img, meta={})
    cam_to_xyz = np.array(
        [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]],
        dtype=np.float32,
    )
    xyz_to_working = np.array(
        [[1.1, -0.1, 0.0], [0.0, 1.0, 0.0], [0.0, -0.1, 1.1]],
        dtype=np.float32,
    )
    effective = (xyz_to_working @ cam_to_xyz).astype(np.float32)

    chain = stage_ccm(
        frame,
        {
            "mode": "chain",
            "cam_to_xyz_matrix": cam_to_xyz.tolist(),
            "xyz_to_working_matrix": xyz_to_working.tolist(),
        },
    ).frame.image
    manual = stage_ccm(frame, {"mode": "manual", "matrix": effective.tolist()}).frame.image
    assert np.allclose(chain, manual, atol=1e-6, rtol=1e-6)


def test_ccm_chain_identity_no_change() -> None:
    img = np.random.default_rng(3).random((4, 4, 3), dtype=np.float32)
    frame = Frame(image=img, meta={})
    identity = np.eye(3, dtype=np.float32).tolist()
    result = stage_ccm(
        frame, {"mode": "chain", "cam_to_xyz_matrix": identity, "xyz_to_working_matrix": identity}
    )
    assert np.allclose(result.frame.image, img)
    assert result.frame.meta["ccm_mode"] == "chain"
    assert np.array(result.frame.meta["ccm"]).shape == (3, 3)


def test_ccm_chain_missing_cam_to_xyz_uses_identity_fallback() -> None:
    img = np.random.default_rng(4).random((4, 4, 3), dtype=np.float32)
    frame = Frame(image=img, meta={})
    result = stage_ccm(frame, {"mode": "chain"})

    resolved = result.metrics["resolved_params"]
    assert resolved["mode"] == "chain"
    assert resolved["cam_to_xyz_source"] == "identity_fallback"
    assert resolved["xyz_to_working_source"] == "constant_xyz_to_lin_srgb_d65"
    assert np.array(resolved["cam_to_xyz_matrix"]).shape == (3, 3)
    assert np.array(resolved["xyz_to_working_matrix"]).shape == (3, 3)
    assert resolved["effective_matrix"] == resolved["xyz_to_working_matrix"]

    assert result.frame.meta["ccm_mode"] == "chain"
    assert result.frame.meta["ccm"] == resolved["effective_matrix"]
    assert result.frame.image.dtype == np.float32
    assert result.frame.image.shape == img.shape
    assert np.all(np.isfinite(result.frame.image))


def test_stats_3a_non_invasive_and_keys() -> None:
    img = np.random.default_rng(1).random((8, 8, 3), dtype=np.float32)
    frame = Frame(image=img, meta={})
    result = stage_stats_3a(frame, {})
    assert np.array_equal(result.frame.image, img)

    stats = result.metrics["stats_3a"]
    assert "ae" in stats and "awb" in stats and "af" in stats
    assert len(stats["ae"]["hist"]) == 64
    assert np.isfinite(stats["af"]["tenengrad"])
