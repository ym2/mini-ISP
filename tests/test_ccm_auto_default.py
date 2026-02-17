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


def test_resolve_ccm_auto_default_non_dng_raw_skips() -> None:
    stage_cfg = {}
    frame_meta = {
        "input_kind": "raw",
        "input_ext": ".nef",
        "source_path": "/tmp/sample.nef",
        "cam_to_xyz_matrix": np.eye(3, dtype=np.float32).tolist(),
        "xyz_to_working_matrix": np.eye(3, dtype=np.float32).tolist(),
    }
    out = _resolve_ccm_stage_params(stage_cfg, frame_meta)
    assert out.get("mode") is None
    assert out["auto_default_applied"] is False
    assert out["auto_default_reason"] == "non_dng_raw"


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
