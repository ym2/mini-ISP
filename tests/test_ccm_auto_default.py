from __future__ import annotations

import numpy as np

from mini_isp.io_utils import Frame
from mini_isp.run import _resolve_ccm_stage_params
from mini_isp.stages import stage_ccm


def test_resolve_ccm_explicit_config_wins() -> None:
    stage_cfg = {"mode": "identity"}
    frame_meta = {
        "input_kind": "raw",
        "input_ext": ".dng",
        "cam_to_xyz_matrix": np.eye(3, dtype=np.float32).tolist(),
        "xyz_to_working_matrix": np.eye(3, dtype=np.float32).tolist(),
    }
    out = _resolve_ccm_stage_params(stage_cfg, frame_meta)
    assert out["mode"] == "identity"
    assert out["auto_default_applied"] is False
    assert out["auto_default_reason"] == "explicit_stage_config"


def test_resolve_ccm_auto_default_dng_chain_when_matrices_available() -> None:
    cam_to_xyz = np.eye(3, dtype=np.float32).tolist()
    xyz_to_working = np.eye(3, dtype=np.float32).tolist()
    stage_cfg = {}
    frame_meta = {
        "input_kind": "raw",
        "input_ext": ".dng",
        "source_path": "/tmp/sample.dng",
        "cam_to_xyz_matrix": cam_to_xyz,
        "xyz_to_working_matrix": xyz_to_working,
        "cam_to_xyz_source": "dng_tags_forward_matrix",
        "xyz_to_working_source": "constant_xyz_d50_to_lin_srgb_d65",
    }
    out = _resolve_ccm_stage_params(stage_cfg, frame_meta)
    assert out["mode"] == "chain"
    assert out["cam_to_xyz_matrix"] == cam_to_xyz
    assert out["xyz_to_working_matrix"] == xyz_to_working
    assert out["cam_to_xyz_source"] == "dng_tags_forward_matrix"
    assert out["xyz_to_working_source"] == "constant_xyz_d50_to_lin_srgb_d65"
    assert out["auto_default_applied"] is True
    assert out["auto_default_reason"] == "applied_from_dng_tags"


def test_resolve_ccm_auto_default_non_dng_prefers_daylight_d65() -> None:
    stage_cfg = {}
    identity = np.eye(3, dtype=np.float32)
    frame_meta = {
        "input_kind": "raw",
        "input_ext": ".nef",
        "source_path": "/tmp/sample.nef",
        "wb_gains": [2.0, 1.0, 0.5],
        "daylight_wb_gains": [4.0, 1.0, 2.0],
        "non_dng_cam_to_xyz_matrix": identity.tolist(),
        "non_dng_cam_to_xyz_source": "rawpy_rgb_xyz_matrix_4x3_mergeg_sum_inv",
        "non_dng_selected_input_variant": "pre_unwb",
        "non_dng_xyz_to_working_matrix_d65": identity.tolist(),
        "non_dng_xyz_to_working_source_d65": "constant_xyz_d65_to_lin_srgb_d65",
        "non_dng_xyz_to_working_matrix_d50adapt": (2.0 * identity).tolist(),
        "non_dng_xyz_to_working_source_d50adapt": "constant_xyz_d50_to_lin_srgb_d65",
    }
    out = _resolve_ccm_stage_params(stage_cfg, frame_meta)
    assert out["mode"] == "chain"
    assert out["auto_default_applied"] is True
    assert out["auto_default_reason"] == "applied_non_dng_meta_default"
    assert out["ccm_source"] == "non_dng_meta_default"
    assert out["non_dng_meta_rule"] == "prefer_pre_unwb_daylight_d65_else_selected_d50adapt"
    assert out["non_dng_meta_input_variant"] == "pre_unwb_daylight"
    assert out["non_dng_meta_wp_variant"] == "d65"
    expected_cam = np.diag([0.25, 1.0, 0.5]).astype(np.float32)
    assert np.allclose(np.asarray(out["cam_to_xyz_matrix"], dtype=np.float32), expected_cam, atol=1e-6)
    assert np.allclose(np.asarray(out["xyz_to_working_matrix"], dtype=np.float32), identity, atol=1e-6)


def test_resolve_ccm_auto_default_dng_missing_matrices_skips_with_reason() -> None:
    stage_cfg = {}
    frame_meta = {
        "input_kind": "raw",
        "input_ext": ".dng",
        "source_path": "/tmp/sample.dng",
        "cam_to_xyz_source": "none",
        "xyz_to_working_source": "none",
        "ccm_auto_reason": "missing_dng_tags",
    }
    out = _resolve_ccm_stage_params(stage_cfg, frame_meta)
    assert out.get("mode") is None
    assert out["auto_default_applied"] is False
    assert out["auto_default_reason"] == "missing_dng_tags"
    assert out["cam_to_xyz_source"] == "none"
    assert out["xyz_to_working_source"] == "none"


def test_resolve_ccm_auto_default_non_dng_fallback_selected_d50adapt() -> None:
    stage_cfg = {}
    identity = np.eye(3, dtype=np.float32)
    frame_meta = {
        "input_kind": "raw",
        "input_ext": ".cr2",
        "source_path": "/tmp/sample.cr2",
        "wb_gains": [2.0, 1.0, 0.5],
        "non_dng_cam_to_xyz_matrix": identity.tolist(),
        "non_dng_cam_to_xyz_source": "rawpy_rgb_xyz_matrix_4x3_mergeg_sum_inv",
        "non_dng_selected_input_variant": "pre_unwb",
        "non_dng_xyz_to_working_matrix_d50adapt": identity.tolist(),
        "non_dng_xyz_to_working_source_d50adapt": "constant_xyz_d50_to_lin_srgb_d65",
    }
    out = _resolve_ccm_stage_params(stage_cfg, frame_meta)
    assert out["mode"] == "chain"
    assert out["auto_default_applied"] is True
    assert out["non_dng_meta_input_variant"] == "selected_input"
    assert out["non_dng_meta_wp_variant"] == "d50adapt"
    expected_cam = np.diag([0.5, 1.0, 2.0]).astype(np.float32)
    assert np.allclose(np.asarray(out["cam_to_xyz_matrix"], dtype=np.float32), expected_cam, atol=1e-6)
    assert np.allclose(np.asarray(out["xyz_to_working_matrix"], dtype=np.float32), identity, atol=1e-6)


def test_resolve_ccm_auto_default_non_dng_missing_matrix_skips_with_reason() -> None:
    stage_cfg = {}
    frame_meta = {
        "input_kind": "raw",
        "input_ext": ".nef",
        "source_path": "/tmp/sample.nef",
        "non_dng_meta_reason": "missing_rgb_xyz_matrix",
    }
    out = _resolve_ccm_stage_params(stage_cfg, frame_meta)
    assert out.get("mode") is None
    assert out["auto_default_applied"] is False
    assert out["auto_default_reason"] == "missing_rgb_xyz_matrix"


def test_ccm_chain_source_overrides_appear_in_debug_resolved_params() -> None:
    img = np.random.default_rng(0).random((2, 2, 3), dtype=np.float32)
    frame = Frame(image=img, meta={})
    identity = np.eye(3, dtype=np.float32).tolist()
    result = stage_ccm(
        frame,
        {
            "mode": "chain",
            "cam_to_xyz_matrix": identity,
            "xyz_to_working_matrix": identity,
            "cam_to_xyz_source": "dng_tags_forward_matrix",
            "xyz_to_working_source": "constant_xyz_d50_to_lin_srgb_d65",
        },
    )
    params = result.metrics["resolved_params"]
    assert params["cam_to_xyz_source"] == "dng_tags_forward_matrix"
    assert params["xyz_to_working_source"] == "constant_xyz_d50_to_lin_srgb_d65"
