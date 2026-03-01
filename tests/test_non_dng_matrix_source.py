from __future__ import annotations

import numpy as np

from mini_isp.io_utils import derive_non_dng_ccm_from_rawpy_matrix


def test_non_dng_matrix_source_rejects_unsupported_shape() -> None:
    bad = np.ones((2, 2), dtype=np.float32)
    out = derive_non_dng_ccm_from_rawpy_matrix(bad, wb_gains_rgb=(2.0, 1.0, 0.5))
    assert out["available"] is False
    assert out["reason"].startswith("unsupported_rgb_xyz_matrix_shape_")


def test_non_dng_matrix_source_prefers_pre_unwb_when_wb_present() -> None:
    matrix = np.diag([1.9, 1.0, 0.54]).astype(np.float32)
    out = derive_non_dng_ccm_from_rawpy_matrix(matrix, wb_gains_rgb=(2.0, 1.0, 0.5))
    assert out["available"] is True
    assert out["selected_input_variant"] == "pre_unwb"
    assert out["cam_to_xyz_source"] == "rawpy_rgb_xyz_matrix_3x3_as_is"
    assert out["selection_policy"] == "wp_error_min_det"
    assert out["candidate_count"] >= 1
    assert out["selected_wp_err_min"] < 0.05


def test_non_dng_matrix_source_uses_as_is_without_wb() -> None:
    m43 = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    out = derive_non_dng_ccm_from_rawpy_matrix(m43, wb_gains_rgb=None)
    assert out["available"] is True
    assert out["selected_input_variant"] == "as_is"
    assert out["selection_policy"] == "wp_error_min_det"
    assert out["candidate_count"] >= 1
