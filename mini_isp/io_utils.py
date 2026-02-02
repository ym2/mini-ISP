from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

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


def _color_desc_str(color_desc: Optional[Any]) -> str:
    if color_desc is None:
        return ""
    if isinstance(color_desc, bytes):
        desc = color_desc.decode(errors="ignore")
    elif hasattr(color_desc, "tobytes"):
        desc = color_desc.tobytes().decode(errors="ignore")
    else:
        desc = str(color_desc)
    return desc.strip()


def derive_cfa_pattern(
    raw_pattern: Optional[np.ndarray], color_desc: Optional[Any], fallback: str
) -> str:
    if raw_pattern is None or color_desc is None:
        return fallback
    pattern = np.asarray(raw_pattern)
    if pattern.shape != (2, 2):
        return fallback
    desc = _color_desc_str(color_desc)
    if not desc:
        return fallback
    try:
        letters = "".join(desc[int(idx)] for idx in pattern.flatten())
    except Exception:
        return fallback
    return letters


def _normalize_wb_gains(cfa_pattern: str, gains: Any) -> Tuple[float, float, float]:
    values = [float(x) for x in gains] if gains is not None else []
    if not values:
        return 1.0, 1.0, 1.0
    if len(values) >= 4 and len(cfa_pattern) == 4:
        r_vals = [values[i] for i, ch in enumerate(cfa_pattern) if ch == "R" and values[i] > 0]
        g_vals = [values[i] for i, ch in enumerate(cfa_pattern) if ch == "G" and values[i] > 0]
        b_vals = [values[i] for i, ch in enumerate(cfa_pattern) if ch == "B" and values[i] > 0]
        r = float(np.mean(r_vals)) if r_vals else None
        g = float(np.mean(g_vals)) if g_vals else None
        b = float(np.mean(b_vals)) if b_vals else None
        if r is None or g is None or b is None:
            return 1.0, 1.0, 1.0
    elif len(values) >= 3:
        r, g, b = values[0], values[1], values[2]
    else:
        r = g = b = values[0]
    g_base = g if abs(g) > 1e-12 else 1.0
    return r / g_base, 1.0, b / g_base


def _normalize_wb_gains_from_desc(desc: str, gains: Any) -> Optional[Tuple[float, float, float]]:
    values = [float(x) for x in gains] if gains is not None else []
    if not desc or not values:
        return None
    if len(values) < len(desc):
        return None
    r_vals = [values[i] for i, ch in enumerate(desc) if ch == "R" and values[i] > 0]
    g_vals = [values[i] for i, ch in enumerate(desc) if ch == "G" and values[i] > 0]
    b_vals = [values[i] for i, ch in enumerate(desc) if ch == "B" and values[i] > 0]
    if not r_vals or not g_vals or not b_vals:
        return None
    r = float(np.mean(r_vals))
    g = float(np.mean(g_vals))
    b = float(np.mean(b_vals))
    g_base = g if abs(g) > 1e-12 else 1.0
    return r / g_base, 1.0, b / g_base


def _select_raw_wb(raw: Any, cfa_pattern: str, color_desc: str) -> Tuple[Tuple[float, float, float], str]:
    camera_wb = getattr(raw, "camera_whitebalance", None)
    if camera_wb is not None:
        gains = _normalize_wb_gains_from_desc(color_desc, camera_wb)
        if gains is None:
            gains = _normalize_wb_gains(cfa_pattern, camera_wb)
        return gains, "camera_whitebalance"
    daylight_wb = getattr(raw, "daylight_whitebalance", None)
    if daylight_wb is not None:
        gains = _normalize_wb_gains_from_desc(color_desc, daylight_wb)
        if gains is None:
            gains = _normalize_wb_gains(cfa_pattern, daylight_wb)
        return gains, "daylight_whitebalance"
    return (1.0, 1.0, 1.0), "unity"


def load_raw_mosaic(path: str, fallback_cfa: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import rawpy  # type: ignore
    except Exception as exc:
        raise RuntimeError("rawpy is required to load RAW/DNG inputs") from exc

    with rawpy.imread(path) as raw:
        mosaic = getattr(raw, "raw_image_visible", None)
        if mosaic is None:
            mosaic = raw.raw_image
        mosaic = np.asarray(mosaic)

        white_level = getattr(raw, "white_level", None)
        black_level = None
        black_level_per_channel = getattr(raw, "black_level_per_channel", None)
        if black_level_per_channel is not None:
            try:
                black_level = float(np.mean(black_level_per_channel))
            except Exception:
                black_level = None
        if black_level is None:
            black_level = float(getattr(raw, "black_level", 0.0))

        bit_depth = None
        if white_level is not None:
            try:
                bit_depth = int(np.ceil(np.log2(float(white_level) + 1.0)))
            except Exception:
                bit_depth = None
        if bit_depth is None:
            if np.issubdtype(mosaic.dtype, np.integer):
                bit_depth = int(np.iinfo(mosaic.dtype).bits)

        color_desc = _color_desc_str(getattr(raw, "color_desc", None))
        cfa_pattern = derive_cfa_pattern(
            getattr(raw, "raw_pattern", None),
            getattr(raw, "color_desc", None),
            fallback_cfa,
        )
        wb_gains, wb_source = _select_raw_wb(raw, cfa_pattern, color_desc)

    meta = {
        "cfa_pattern": cfa_pattern,
        "black_level": float(black_level) if black_level is not None else 0.0,
        "white_level": float(white_level) if white_level is not None else None,
        "bit_depth": bit_depth,
        "wb_gains": [float(x) for x in wb_gains],
        "wb_source": wb_source,
    }
    return mosaic, meta


def normalize_raw_mosaic(raw: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    eps = 1e-6
    denom = max(float(white_level) - float(black_level), eps)
    norm = (raw.astype(np.float32) - float(black_level)) / denom
    return np.clip(norm, 0.0, 1.0).astype(np.float32)
