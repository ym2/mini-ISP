from __future__ import annotations

import numpy as np

from mini_isp.io_utils import Frame
from mini_isp.stages import stage_color_adjust, stage_oetf_encode, stage_sharpen, stage_tone


def test_tone_methods_monotonic() -> None:
    x = np.linspace(0.0, 10.0, 16, dtype=np.float32)
    img = np.stack([x, x, x], axis=1).reshape(1, -1, 3)
    frame = Frame(image=img, meta={})
    out_reinhard = stage_tone(frame, {"method": "reinhard"}).frame.image
    out_filmic = stage_tone(frame, {"method": "filmic"}).frame.image
    assert out_reinhard.shape == img.shape
    assert out_filmic.shape == img.shape
    assert np.all(np.diff(out_reinhard[0, :, 0]) >= 0)
    assert np.all(np.diff(out_filmic[0, :, 0]) >= 0)


def test_color_adjust_identity_noop() -> None:
    img = np.random.default_rng(0).random((4, 4, 3), dtype=np.float32)
    frame = Frame(image=img, meta={})
    out = stage_color_adjust(frame, {"method": "identity"}).frame.image
    assert np.array_equal(out, img)


def test_color_adjust_chroma_scale_preserves_luma() -> None:
    img = np.array([[[0.2, 0.5, 0.8]]], dtype=np.float32)
    frame = Frame(image=img, meta={})
    out = stage_color_adjust(frame, {"method": "chroma_scale_lrgb", "sat_scale": 2.0}).frame.image
    y_in = 0.2126 * img[0, 0, 0] + 0.7152 * img[0, 0, 1] + 0.0722 * img[0, 0, 2]
    y_out = 0.2126 * out[0, 0, 0] + 0.7152 * out[0, 0, 1] + 0.0722 * out[0, 0, 2]
    assert np.isclose(y_in, y_out, atol=1e-6)


def test_sharpen_increases_edge_contrast() -> None:
    img = np.zeros((8, 8, 3), dtype=np.float32)
    img[:, 4:, :] = 1.0
    frame = Frame(image=img, meta={})
    out = stage_sharpen(frame, {"method": "unsharp_mask", "sigma": 1.0, "amount": 1.0}).frame.image
    # Measure gradient magnitude along the edge
    grad_before = np.abs(np.diff(img[:, :, 0], axis=1)).mean()
    grad_after = np.abs(np.diff(out[:, :, 0], axis=1)).mean()
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert grad_after >= grad_before


def test_oetf_encode_uint8_range() -> None:
    img = np.array([[[0.0, 0.5, 2.0]]], dtype=np.float32)
    frame = Frame(image=img, meta={})
    result = stage_oetf_encode(frame, {"oetf": "srgb"})
    out = result.frame.image
    assert out.dtype == np.uint8
    assert int(out.min()) >= 0
    assert int(out.max()) <= 255
