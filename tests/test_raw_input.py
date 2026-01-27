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
