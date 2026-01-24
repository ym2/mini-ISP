from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


@dataclass
class Frame:
    image: np.ndarray
    meta: Dict[str, Any]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_png_as_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def save_png(path: str, image_u8: np.ndarray) -> None:
    Image.fromarray(image_u8).save(path)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    # Basic sRGB OETF approximation for linear values in [0, 1]
    a = 0.055
    linear = np.clip(linear, 0.0, 1.0)
    encoded = np.where(linear <= 0.0031308, linear * 12.92, (1 + a) * np.power(linear, 1 / 2.4) - a)
    return encoded


def to_display_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    encoded = srgb_encode(image)
    return np.clip(encoded * 255.0 + 0.5, 0, 255).astype(np.uint8)


def downscale_max_side(image_u8: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image_u8.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image_u8
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return np.asarray(Image.fromarray(image_u8).resize((new_w, new_h), Image.BILINEAR))


def roi_crop(image_u8: np.ndarray, xywh_norm: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = image_u8.shape[:2]
    x, y, rw, rh = xywh_norm
    x0 = int(round(np.clip(x, 0.0, 1.0) * w))
    y0 = int(round(np.clip(y, 0.0, 1.0) * h))
    x1 = int(round(np.clip(x + rw, 0.0, 1.0) * w))
    y1 = int(round(np.clip(y + rh, 0.0, 1.0) * h))
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return image_u8[y0:y1, x0:x1]


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_yaml(path: str, data: Dict[str, Any]) -> None:
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    except Exception:
        # Fallback: write JSON when PyYAML is unavailable
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=False)
