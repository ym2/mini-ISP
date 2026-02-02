from __future__ import annotations

from dataclasses import dataclass

from mini_isp.io_utils import _normalize_wb_gains, _normalize_wb_gains_from_desc, _select_raw_wb


@dataclass
class _DummyRaw:
    camera_whitebalance: list[float] | None = None
    daylight_whitebalance: list[float] | None = None


def test_select_raw_wb_camera_priority() -> None:
    raw = _DummyRaw(camera_whitebalance=[2.0, 1.0, 1.0, 1.0])
    gains, source = _select_raw_wb(raw, "RGGB", "RGBG")
    assert source == "camera_whitebalance"
    assert gains == (2.0, 1.0, 1.0)


def test_select_raw_wb_daylight_fallback() -> None:
    raw = _DummyRaw(daylight_whitebalance=[1.0, 2.0, 4.0, 2.0])
    gains, source = _select_raw_wb(raw, "RGGB", "RGBG")
    assert source == "daylight_whitebalance"
    assert gains == (0.5, 1.0, 2.0)


def test_select_raw_wb_unity_fallback() -> None:
    raw = _DummyRaw()
    gains, source = _select_raw_wb(raw, "RGGB", "RGBG")
    assert source == "unity"
    assert gains == (1.0, 1.0, 1.0)


def test_normalize_wb_gains_len3() -> None:
    gains = _normalize_wb_gains("RGGB", [2.0, 4.0, 8.0])
    assert gains == (0.5, 1.0, 2.0)


def test_normalize_wb_gains_from_desc_ignores_zero() -> None:
    gains = _normalize_wb_gains_from_desc("RGBG", [1.972656, 1.0, 1.671875, 0.0])
    assert gains == (1.972656, 1.0, 1.671875)
