import numpy as np

from mini_isp.io_utils import Frame
from mini_isp.stages import stage_sharpen


def _step_edge_image(width: int = 64, height: int = 48) -> np.ndarray:
    left = np.full((height, width // 2, 3), 0.2, dtype=np.float32)
    right = np.full((height, width - width // 2, 3), 0.8, dtype=np.float32)
    image = np.concatenate([left, right], axis=1)
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.002, size=image.shape).astype(np.float32)
    return image + noise


def _edge_metrics(img: np.ndarray, ref: np.ndarray, edge_col: int) -> tuple[float, float]:
    y = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    y_ref = 0.2126 * ref[:, :, 0] + 0.7152 * ref[:, :, 1] + 0.0722 * ref[:, :, 2]
    grad = np.abs(np.diff(y, axis=1))
    band_start = max(0, edge_col - 5)
    band_end = min(grad.shape[1], edge_col + 5)
    edge_strength = float(np.mean(grad[:, band_start:band_end]))

    band_start_img = max(0, edge_col - 5)
    band_end_img = min(img.shape[1], edge_col + 5)
    out_band = y[:, band_start_img:band_end_img]
    in_band = y_ref[:, band_start_img:band_end_img]
    overshoot_hi = float(np.max(out_band) - np.max(in_band))
    overshoot_lo = float(np.min(in_band) - np.min(out_band))
    overshoot = max(0.0, overshoot_hi, overshoot_lo)
    return edge_strength, overshoot


def test_unsharp_mask_tuned_reduces_overshoot() -> None:
    image = _step_edge_image()
    frame = Frame(image=image, meta={})
    edge_col = image.shape[1] // 2

    baseline = stage_sharpen(
        frame,
        {"method": "unsharp_mask", "sigma": 1.0, "amount": 0.5, "threshold": 0.0},
    ).frame.image
    tuned = stage_sharpen(
        frame,
        {
            "method": "unsharp_mask_tuned",
            "sigma": 1.0,
            "amount": 0.4,
            "threshold": 0.01,
            "luma_only": True,
            "gate_mode": "soft",
        },
    ).frame.image

    assert baseline.shape == image.shape
    assert tuned.shape == image.shape
    assert baseline.dtype == np.float32
    assert tuned.dtype == np.float32
    assert np.isfinite(baseline).all()
    assert np.isfinite(tuned).all()

    base_strength, base_overshoot = _edge_metrics(baseline, image, edge_col)
    tuned_strength, tuned_overshoot = _edge_metrics(tuned, image, edge_col)

    tol = 0.1
    assert tuned_strength >= base_strength * (1.0 - tol)
    assert tuned_overshoot <= base_overshoot + 1e-6
