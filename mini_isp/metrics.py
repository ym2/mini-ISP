from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

from .io_utils import load_png_as_rgb


def build_preview_for_metrics(stage_root: str, config: Dict[str, Any]) -> Optional[np.ndarray]:
    preview_path = os.path.join(stage_root, "preview.png")
    if not os.path.exists(preview_path):
        return None
    return load_png_as_rgb(preview_path).astype(np.float32)


def _luma(image: np.ndarray) -> np.ndarray:
    return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]


def _chroma_means(image: np.ndarray) -> Dict[str, float]:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    return {
        "mean_r_minus_g": float(np.mean(r - g)),
        "mean_b_minus_g": float(np.mean(b - g)),
    }


def _basic_stats(image: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "p01": float(np.percentile(image, 1.0)),
        "p99": float(np.percentile(image, 99.0)),
    }


def _clip_pct(image: np.ndarray, lo: float = 0.0, hi: float = 255.0) -> float:
    return float(np.mean((image <= lo) | (image >= hi)) * 100.0)


def _diff_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = a - b
    l1 = float(np.mean(np.abs(diff)))
    l2 = float(np.mean(diff * diff))
    if l2 > 0:
        psnr = 20.0 * np.log10(255.0 / np.sqrt(l2))
    else:
        psnr = float("inf")
    return {"l1": l1, "l2": l2, "psnr": float(psnr)}


def _false_color_map(image: np.ndarray) -> np.ndarray:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    chroma = np.sqrt((r - g) ** 2 + (b - g) ** 2)
    chroma = np.clip(chroma / 255.0, 0.0, 1.0)
    return np.stack([chroma, chroma, chroma], axis=2)


def _zipper_proxy(image: np.ndarray) -> np.ndarray:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    chroma = np.abs(r - g) + np.abs(b - g)
    # High-frequency proxy via simple Laplacian
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    padded = np.pad(chroma, 1, mode="edge")
    out = np.zeros_like(chroma)
    for y in range(chroma.shape[0]):
        for x in range(chroma.shape[1]):
            patch = padded[y : y + 3, x : x + 3]
            out[y, x] = np.sum(patch * kernel)
    out = np.abs(out)
    out = np.clip(out / (np.max(out) + 1e-6), 0.0, 1.0)
    return np.stack([out, out, out], axis=2)


def _halo_proxy(image: np.ndarray) -> np.ndarray:
    luma = _luma(image)
    # Edge overshoot proxy: gradient magnitude
    gx = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1]))
    gy = np.abs(np.diff(luma, axis=0, prepend=luma[:1, :]))
    mag = np.clip((gx + gy) / (np.max(gx + gy) + 1e-6), 0.0, 1.0)
    return np.stack([mag, mag, mag], axis=2)


def _save_png(path: str, image_f32: np.ndarray) -> None:
    import PIL.Image

    img = np.clip(image_f32 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    PIL.Image.fromarray(img).save(path)


def emit_metrics_and_diagnostics(
    stage_root: str,
    preview: np.ndarray,
    prev_preview: Optional[np.ndarray],
    enable_diagnostics: bool,
    metrics_out: str = "extra",
) -> None:
    extra_root = os.path.join(stage_root, "extra")
    if metrics_out == "stage_root":
        os.makedirs(stage_root, exist_ok=True)
        metrics_path = os.path.join(stage_root, "metrics.json")
        diff_path = os.path.join(stage_root, "diff_metrics.json")
    else:
        os.makedirs(extra_root, exist_ok=True)
        metrics_path = os.path.join(extra_root, "metrics.json")
        diff_path = os.path.join(extra_root, "diff_metrics.json")
    metrics: Dict[str, Any] = {}
    metrics.update(_basic_stats(preview))
    metrics["clip_pct"] = _clip_pct(preview)
    metrics["luma_mean"] = float(np.mean(_luma(preview)))
    metrics.update(_chroma_means(preview))
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if prev_preview is not None:
        diff = _diff_metrics(preview, prev_preview)
        with open(diff_path, "w", encoding="utf-8") as f:
            json.dump(diff, f, indent=2)

    if enable_diagnostics:
        diag_root = os.path.join(extra_root, "diagnostics")
        os.makedirs(diag_root, exist_ok=True)
        _save_png(os.path.join(diag_root, "false_color.png"), _false_color_map(preview))
        _save_png(os.path.join(diag_root, "zipper.png"), _zipper_proxy(preview))
        _save_png(os.path.join(diag_root, "halo.png"), _halo_proxy(preview))
