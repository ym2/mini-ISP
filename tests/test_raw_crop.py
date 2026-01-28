from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mini_isp.tools.raw_crop import _shift_cfa_pattern, _validate_bounds, run_crop


def test_shift_cfa_pattern_parity() -> None:
    assert _shift_cfa_pattern("RGGB", 0, 0) == "RGGB"
    assert _shift_cfa_pattern("RGGB", 1, 0) == "GRBG"
    assert _shift_cfa_pattern("RGGB", 0, 1) == "GBRG"
    assert _shift_cfa_pattern("RGGB", 1, 1) == "BGGR"


def test_validate_bounds() -> None:
    _validate_bounds(0, 0, 2, 2, (4, 4))
    with pytest.raises(ValueError):
        _validate_bounds(3, 0, 2, 2, (4, 4))


def test_crop_deterministic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mosaic = np.arange(16, dtype=np.uint16).reshape(4, 4)

    def fake_loader(path: str, fallback: str):
        return mosaic, {
            "cfa_pattern": "RGGB",
            "black_level": 0.0,
            "white_level": 15.0,
            "bit_depth": 4,
        }

    monkeypatch.setattr("mini_isp.tools.raw_crop.load_raw_mosaic", fake_loader)

    out_dir = tmp_path / "crop"
    run_crop("fake.dng", str(out_dir), 1, 1, 2, 2, "float32", False)
    crop = np.load(out_dir / "crop.npy")
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert crop.shape == (2, 2)
    assert crop.dtype == np.float32
    assert meta["cfa_pattern"] == "BGGR"

    # Determinism check
    out_dir2 = tmp_path / "crop2"
    run_crop("fake.dng", str(out_dir2), 1, 1, 2, 2, "float32", False)
    crop2 = np.load(out_dir2 / "crop.npy")
    assert np.allclose(crop, crop2)


def test_rawpy_optional(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rawpy = pytest.importorskip("rawpy")

    class DummyRaw:
        def __init__(self) -> None:
            self.raw_image_visible = np.arange(16, dtype=np.uint16).reshape(4, 4)
            self.raw_image = self.raw_image_visible
            self.white_level = 15
            self.black_level_per_channel = [0, 0, 0, 0]
            self.raw_pattern = np.array([[0, 1], [1, 2]], dtype=np.uint8)
            self.color_desc = b"RGBG"

        def __enter__(self) -> "DummyRaw":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_imread(path: str):
        return DummyRaw()

    monkeypatch.setattr(rawpy, "imread", fake_imread)
    out_dir = tmp_path / "crop_rawpy"
    run_crop("fake.dng", str(out_dir), 0, 0, 2, 2, "uint16", True)
    crop = np.load(out_dir / "crop.npy")
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert crop.shape == (2, 2)
    assert crop.dtype == np.uint16
    assert meta["cfa_pattern"] == "RGGB"
