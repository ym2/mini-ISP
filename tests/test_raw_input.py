from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from mini_isp.io_utils import derive_cfa_pattern, normalize_raw_mosaic, load_raw_mosaic
from mini_isp.run import run_pipeline, DEFAULT_CONFIG


def test_normalize_raw_mosaic_and_cfa_fallback() -> None:
    raw = np.array([[50, 150], [250, 400]], dtype=np.uint16)
    norm = normalize_raw_mosaic(raw, black_level=100.0, white_level=300.0)
    assert norm.dtype == np.float32
    assert norm.shape == raw.shape
    assert float(np.min(norm)) >= 0.0
    assert float(np.max(norm)) <= 1.0

    cfa = derive_cfa_pattern(None, None, "RGGB")
    assert cfa == "RGGB"


def test_rawpy_loader_and_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rawpy = pytest.importorskip("rawpy")

    class DummyRaw:
        def __init__(self) -> None:
            self.raw_image_visible = np.array([[100, 200], [300, 400]], dtype=np.uint16)
            self.raw_image = self.raw_image_visible
            self.white_level = 4095
            self.black_level_per_channel = [64, 64, 64, 64]
            self.raw_pattern = np.array([[0, 1], [1, 2]], dtype=np.uint8)
            self.color_desc = b"RGBG"

        def __enter__(self) -> "DummyRaw":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_imread(path: str) -> DummyRaw:
        return DummyRaw()

    monkeypatch.setattr(rawpy, "imread", fake_imread)

    mosaic, meta = load_raw_mosaic("fake.dng", "RGGB")
    assert meta["cfa_pattern"] == "RGGB"
    assert meta["bit_depth"] == 12
    norm = normalize_raw_mosaic(mosaic, meta["black_level"], meta["white_level"])
    assert norm.dtype == np.float32
    assert norm.shape == mosaic.shape

    config = DEFAULT_CONFIG.copy()
    config["input"] = dict(DEFAULT_CONFIG["input"])
    config["output"] = dict(DEFAULT_CONFIG["output"])
    config["input"]["path"] = "fake.dng"
    config["output"]["dir"] = str(tmp_path)
    config["output"]["name"] = "raw_run"

    run_pipeline(config)
    manifest_path = tmp_path / "raw_run" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "cfa_pattern" in manifest["input"]
    assert "bit_depth" in manifest["input"]
    assert "black_level" in manifest["input"]
    assert "white_level" in manifest["input"]


def test_pipeline_with_bggr_cfa(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_loader(path: str, fallback: str):
        mosaic = np.full((4, 4), 100, dtype=np.uint16)
        meta = {
            "cfa_pattern": "BGGR",
            "black_level": 0.0,
            "white_level": 255.0,
            "bit_depth": 8,
        }
        return mosaic, meta

    monkeypatch.setattr("mini_isp.run.load_raw_mosaic", fake_loader)
    config = DEFAULT_CONFIG.copy()
    config["input"] = dict(DEFAULT_CONFIG["input"])
    config["output"] = dict(DEFAULT_CONFIG["output"])
    config["input"]["path"] = "fake.dng"
    config["output"]["dir"] = str(tmp_path)
    config["output"]["name"] = "raw_bggr"

    run_root = run_pipeline(config)
    manifest_path = Path(run_root) / "manifest.json"
    assert manifest_path.exists()


def test_load_raw_mosaic_dng_adds_ccm_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    rawpy = pytest.importorskip("rawpy")

    class DummyRaw:
        def __init__(self) -> None:
            self.raw_image_visible = np.array([[100, 200], [300, 400]], dtype=np.uint16)
            self.raw_image = self.raw_image_visible
            self.white_level = 4095
            self.black_level_per_channel = [64, 64, 64, 64]
            self.raw_pattern = np.array([[0, 1], [1, 2]], dtype=np.uint8)
            self.color_desc = b"RGBG"

        def __enter__(self) -> "DummyRaw":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_imread(path: str) -> DummyRaw:
        return DummyRaw()

    monkeypatch.setattr(rawpy, "imread", fake_imread)
    monkeypatch.setattr(
        "mini_isp.io_utils.derive_dng_ccm_from_file",
        lambda _path: {
            "available": True,
            "cam_to_xyz_matrix": np.eye(3, dtype=np.float32).tolist(),
            "cam_to_xyz_source": "dng_tags_forward_matrix",
            "xyz_to_working_matrix": np.eye(3, dtype=np.float32).tolist(),
            "xyz_to_working_source": "constant_xyz_d50_to_lin_srgb_d65",
        },
    )

    _mosaic, meta = load_raw_mosaic("fake.dng", "RGGB")
    assert meta["cam_to_xyz_source"] == "dng_tags_forward_matrix"
    assert meta["xyz_to_working_source"] == "constant_xyz_d50_to_lin_srgb_d65"
    assert np.array(meta["cam_to_xyz_matrix"], dtype=np.float32).shape == (3, 3)
    assert np.array(meta["xyz_to_working_matrix"], dtype=np.float32).shape == (3, 3)
    assert meta["ccm_auto_reason"] == "dng_tags_available"


def test_load_raw_mosaic_rejects_non_bayer_dng(monkeypatch: pytest.MonkeyPatch) -> None:
    rawpy = pytest.importorskip("rawpy")

    class DummyRaw:
        def __init__(self) -> None:
            # Simulate an RGB DNG payload (H, W, C) that is not a Bayer mosaic.
            self.raw_image_visible = np.zeros((8, 12, 4), dtype=np.uint16)
            self.raw_image = self.raw_image_visible
            self.white_level = 4095
            self.black_level_per_channel = [64, 64, 64, 64]
            self.raw_pattern = np.array([[0, 1], [1, 2]], dtype=np.uint8)
            self.color_desc = b"RGBG"

        def __enter__(self) -> "DummyRaw":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_imread(path: str) -> DummyRaw:
        return DummyRaw()

    monkeypatch.setattr(rawpy, "imread", fake_imread)

    with pytest.raises(ValueError, match="not a 2D Bayer mosaic"):
        load_raw_mosaic("fake_rgb.dng", "RGGB")


def test_load_raw_mosaic_non_dng_adds_non_dng_ccm_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    rawpy = pytest.importorskip("rawpy")

    class DummyRaw:
        def __init__(self) -> None:
            self.raw_image_visible = np.array([[100, 200], [300, 400]], dtype=np.uint16)
            self.raw_image = self.raw_image_visible
            self.white_level = 4095
            self.black_level_per_channel = [64, 64, 64, 64]
            self.raw_pattern = np.array([[0, 1], [1, 2]], dtype=np.uint8)
            self.color_desc = b"RGBG"
            self.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
            self.daylight_whitebalance = [2.2, 1.0, 1.4, 1.0]
            self.rgb_xyz_matrix = np.array(
                [
                    [0.8, 0.1, 0.1],
                    [0.1, 0.9, 0.0],
                    [0.0, 0.1, 0.9],
                    [0.1, 0.8, 0.1],
                ],
                dtype=np.float32,
            )

        def __enter__(self) -> "DummyRaw":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_imread(path: str) -> DummyRaw:
        return DummyRaw()

    monkeypatch.setattr(rawpy, "imread", fake_imread)

    _mosaic, meta = load_raw_mosaic("fake.nef", "RGGB")
    assert str(meta["non_dng_cam_to_xyz_source"]).startswith("rawpy_rgb_xyz_matrix_4x3_")
    assert meta["non_dng_selected_input_variant"] in ("pre_unwb", "as_is")
    assert meta["non_dng_cam_to_xyz_selection_policy"] == "wp_error_min_det"
    assert isinstance(meta["non_dng_cam_to_xyz_selected_source_variant"], str)
    assert isinstance(meta["non_dng_cam_to_xyz_candidate_count"], int)
    assert meta["non_dng_cam_to_xyz_candidate_count"] >= 1
    assert np.array(meta["non_dng_cam_to_xyz_matrix"], dtype=np.float32).shape == (3, 3)
    assert np.array(meta["non_dng_xyz_to_working_matrix_d65"], dtype=np.float32).shape == (3, 3)
    assert np.array(meta["non_dng_xyz_to_working_matrix_d50adapt"], dtype=np.float32).shape == (3, 3)
    assert meta["non_dng_meta_reason"] == "non_dng_meta_available"
    assert meta["ccm_auto_reason"] == "non_dng_meta_available"
    assert np.array(meta["daylight_wb_gains"], dtype=np.float32).shape == (3,)
