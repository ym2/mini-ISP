from __future__ import annotations

import numpy as np

from mini_isp.io_utils import derive_dng_ccm_from_exif_metadata


def _flat(mat: np.ndarray) -> list[float]:
    return [float(x) for x in mat.reshape(-1)]


def test_dng_ccm_prefers_forward_matrix_when_available() -> None:
    fm1 = np.eye(3, dtype=np.float32)
    fm2 = 3.0 * np.eye(3, dtype=np.float32)
    meta = {
        "ForwardMatrix1": _flat(fm1),
        "ForwardMatrix2": _flat(fm2),
    }
    out = derive_dng_ccm_from_exif_metadata(meta)
    assert out["available"] is True
    assert out["cam_to_xyz_source"] == "dng_tags_forward_matrix"
    got = np.asarray(out["cam_to_xyz_matrix"], dtype=np.float32)
    assert np.allclose(got, 2.0 * np.eye(3, dtype=np.float32), atol=1e-6)
    xyz_to_working = np.asarray(out["xyz_to_working_matrix"], dtype=np.float32)
    assert xyz_to_working.shape == (3, 3)
    assert np.all(np.isfinite(xyz_to_working))


def test_dng_ccm_color_matrix_native_chain_fallback() -> None:
    cm_identity = np.eye(3, dtype=np.float32)
    meta = {
        "ColorMatrix1": _flat(cm_identity),
        "ColorMatrix2": _flat(cm_identity),
    }
    out = derive_dng_ccm_from_exif_metadata(meta)
    assert out["available"] is True
    assert out["cam_to_xyz_source"] == "dng_tags_synthesized_native_chain_fm"
    got = np.asarray(out["cam_to_xyz_matrix"], dtype=np.float32)
    expected = np.diag(np.array([0.96422, 1.0, 0.82521], dtype=np.float32))
    assert np.allclose(got, expected, atol=1e-6)


def test_dng_ccm_unavailable_when_no_usable_matrices() -> None:
    out = derive_dng_ccm_from_exif_metadata({})
    assert out["available"] is False
    assert out["reason"] == "no_usable_dng_matrices"


def test_dng_ccm_rejects_invalid_matrix_values() -> None:
    meta = {"ForwardMatrix1": [0.0] * 9}
    out = derive_dng_ccm_from_exif_metadata(meta)
    assert out["available"] is False
    assert out["reason"] == "invalid_dng_matrix"
