from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Callable, Tuple, Protocol

import numpy as np

from .io_utils import Frame


@dataclass
class StageResult:
    frame: Frame
    metrics: Dict[str, Any]


class StageInterface(Protocol):
    name: str
    display_name: str

    def run(self, frame: Frame, params: Dict[str, Any]) -> StageResult:
        ...


@dataclass
class Stage:
    name: str
    display_name: str
    func: Callable[[Frame, Dict[str, Any]], StageResult]

    def run(self, frame: Frame, params: Dict[str, Any]) -> StageResult:
        return self.func(frame, params)


def _copy_frame(frame: Frame) -> Frame:
    return Frame(image=np.array(frame.image, copy=True), meta=dict(frame.meta))


def stage_raw_norm(frame: Frame, params: Dict[str, Any]) -> StageResult:
    # Treat input as RGB PNG and map to a pseudo-RAW mosaic for v0.1 bootstrap
    image = frame.image.astype(np.float32) / 255.0
    if image.ndim == 3:
        # Luma for pseudo-mosaic intensity
        luma = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    else:
        luma = image
    h, w = luma.shape
    mosaic = np.zeros((h, w), dtype=np.float32)
    mosaic[0::2, 0::2] = luma[0::2, 0::2]  # R
    mosaic[0::2, 1::2] = luma[0::2, 1::2]  # G
    mosaic[1::2, 0::2] = luma[1::2, 0::2]  # G
    mosaic[1::2, 1::2] = luma[1::2, 1::2]  # B
    mosaic = np.clip(mosaic, 0.0, 1.0).astype(np.float32)

    meta = dict(frame.meta)
    meta.setdefault("cfa_pattern", params.get("cfa_pattern", "RGGB"))
    meta.setdefault("black_level", params.get("black_level", 0.0))
    meta.setdefault("white_level", params.get("white_level", 1.0))

    metrics = {
        "dtype": str(mosaic.dtype),
        "shape": [int(mosaic.shape[0]), int(mosaic.shape[1])],
        "min": float(np.min(mosaic)),
        "max": float(np.max(mosaic)),
        "p01": float(np.percentile(mosaic, 1.0)),
        "p99": float(np.percentile(mosaic, 99.0)),
    }
    return StageResult(frame=Frame(image=mosaic, meta=meta), metrics=metrics)


def stage_stub_identity(frame: Frame, params: Dict[str, Any]) -> StageResult:
    return StageResult(frame=_copy_frame(frame), metrics={})


def inject_defects_for_test(
    mosaic: np.ndarray, coords: Tuple[Tuple[int, int], ...], values: Tuple[float, ...]
) -> np.ndarray:
    """Tests-only helper: inject deterministic defects at known coordinates."""
    if len(coords) != len(values):
        raise ValueError("coords and values length must match")
    out = np.array(mosaic, copy=True)
    h, w = out.shape[:2]
    for (y, x), value in zip(coords, values):
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = value
    return out


def stage_dpc(frame: Frame, params: Dict[str, Any]) -> StageResult:
    # Median-of-neighbors (3x3 excluding center), edge-clamped borders
    image = frame.image.astype(np.float32)
    if image.ndim != 2:
        return StageResult(frame=_copy_frame(frame), metrics={"warning": "dpc expects RAW mosaic"})
    threshold = float(params.get("threshold", 0.2))
    padded = np.pad(image, 1, mode="edge")
    h, w = image.shape
    neighbors = [
        padded[0:h, 0:w],
        padded[0:h, 1 : w + 1],
        padded[0:h, 2 : w + 2],
        padded[1 : h + 1, 0:w],
        padded[1 : h + 1, 2 : w + 2],
        padded[2 : h + 2, 0:w],
        padded[2 : h + 2, 1 : w + 1],
        padded[2 : h + 2, 2 : w + 2],
    ]
    median = np.median(np.stack(neighbors, axis=0), axis=0).astype(np.float32)
    diff = np.abs(image - median)
    mask = diff > threshold
    corrected = np.where(mask, median, image).astype(np.float32)
    metrics = {
        "n_fixed": int(np.sum(mask)),
        "threshold": threshold,
        "min_before": float(np.min(image)),
        "max_before": float(np.max(image)),
        "min_after": float(np.min(corrected)),
        "max_after": float(np.max(corrected)),
    }
    return StageResult(frame=Frame(image=corrected, meta=dict(frame.meta)), metrics=metrics)


def stage_lsc(frame: Frame, params: Dict[str, Any]) -> StageResult:
    image = frame.image.astype(np.float32)
    if image.ndim != 2:
        return StageResult(frame=_copy_frame(frame), metrics={"warning": "lsc expects RAW mosaic"})
    gain_cap = float(params.get("gain_cap", 2.0))
    k = float(params.get("k", 0.5))
    h, w = image.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = float(np.max(r)) if np.max(r) > 0 else 1.0
    gain = 1.0 + k * (r / r_max) ** 2
    gain = np.minimum(gain, gain_cap).astype(np.float32)
    corrected = (image * gain).astype(np.float32)
    metrics = {
        "gain_min": float(np.min(gain)),
        "gain_max": float(np.max(gain)),
        "gain_mean": float(np.mean(gain)),
        "gain_cap": gain_cap,
        "k": k,
        "clipped": False,
    }
    return StageResult(frame=Frame(image=corrected, meta=dict(frame.meta)), metrics=metrics)


def stage_wb_gains(frame: Frame, params: Dict[str, Any]) -> StageResult:
    image = frame.image.astype(np.float32)
    if image.ndim != 2:
        return StageResult(frame=_copy_frame(frame), metrics={"warning": "wb_gains expects RAW mosaic"})
    meta = dict(frame.meta)
    cfa = meta.get("cfa_pattern", "RGGB")
    if cfa != "RGGB":
        raise ValueError(f"wb_gains only supports RGGB in v0.1 (got {cfa})")
    gains = params.get("wb_gains", params.get("gains", [1.0, 1.0, 1.0]))
    r_gain, g_gain, b_gain = [float(x) for x in gains]
    out = np.array(image, copy=True, dtype=np.float32)
    out[0::2, 0::2] *= r_gain  # R
    out[0::2, 1::2] *= g_gain  # Gr
    out[1::2, 0::2] *= g_gain  # Gb
    out[1::2, 1::2] *= b_gain  # B
    meta["wb_gains"] = [r_gain, g_gain, b_gain]
    meta["wb_applied"] = True
    metrics = {
        "wb_gains": [r_gain, g_gain, b_gain],
        "wb_applied": True,
        "min_after": float(np.min(out)),
        "max_after": float(np.max(out)),
    }
    return StageResult(frame=Frame(image=out, meta=meta), metrics=metrics)


def _demosaic_bilinear_rggb(mosaic: np.ndarray) -> np.ndarray:
    mosaic = mosaic.astype(np.float32)
    h, w = mosaic.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r_mask = (yy % 2 == 0) & (xx % 2 == 0)
    b_mask = (yy % 2 == 1) & (xx % 2 == 1)
    g1_mask = (yy % 2 == 0) & (xx % 2 == 1)  # Gr
    g2_mask = (yy % 2 == 1) & (xx % 2 == 0)  # Gb

    padded = np.pad(mosaic, 1, mode="edge")
    c = padded[1:-1, 1:-1]
    n = padded[0:-2, 1:-1]
    s = padded[2:, 1:-1]
    wv = padded[1:-1, 0:-2]
    e = padded[1:-1, 2:]
    nw = padded[0:-2, 0:-2]
    ne = padded[0:-2, 2:]
    sw = padded[2:, 0:-2]
    se = padded[2:, 2:]

    r = np.zeros_like(mosaic, dtype=np.float32)
    g = np.zeros_like(mosaic, dtype=np.float32)
    b = np.zeros_like(mosaic, dtype=np.float32)

    r[r_mask] = c[r_mask]
    b[b_mask] = c[b_mask]
    g[g1_mask | g2_mask] = c[g1_mask | g2_mask]

    # R at B sites, B at R sites
    r[b_mask] = 0.25 * (nw[b_mask] + ne[b_mask] + sw[b_mask] + se[b_mask])
    b[r_mask] = 0.25 * (nw[r_mask] + ne[r_mask] + sw[r_mask] + se[r_mask])

    # G at R/B sites
    g[r_mask] = 0.25 * (n[r_mask] + s[r_mask] + wv[r_mask] + e[r_mask])
    g[b_mask] = 0.25 * (n[b_mask] + s[b_mask] + wv[b_mask] + e[b_mask])

    # R/B at green sites (directional)
    r[g1_mask] = 0.5 * (wv[g1_mask] + e[g1_mask])
    b[g1_mask] = 0.5 * (n[g1_mask] + s[g1_mask])
    r[g2_mask] = 0.5 * (n[g2_mask] + s[g2_mask])
    b[g2_mask] = 0.5 * (wv[g2_mask] + e[g2_mask])

    return np.stack([r, g, b], axis=2).astype(np.float32)


def stage_demosaic(frame: Frame, params: Dict[str, Any]) -> StageResult:
    image = frame.image.astype(np.float32)
    if image.ndim != 2:
        return StageResult(frame=_copy_frame(frame), metrics={"warning": "demosaic expects RAW mosaic"})
    method = str(params.get("method", "bilinear")).lower()
    clip_applied = False
    clip_range = None
    if method == "bilinear":
        rgb = _demosaic_bilinear_rggb(image)
    elif method == "malvar":
        raise NotImplementedError("demosaic method 'malvar' is not implemented in v0.1")
    else:
        raise ValueError(f"Unknown demosaic method: {method}")

    if params.get("clip", False):
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        clip_applied = True
        clip_range = [0.0, 1.0]

    metrics = {
        "method": method,
        "clip_applied": clip_applied,
        "clip_range": clip_range,
        "min": float(np.min(rgb)),
        "max": float(np.max(rgb)),
        "p01": float(np.percentile(rgb, 1.0)),
        "p99": float(np.percentile(rgb, 99.0)),
    }
    return StageResult(frame=Frame(image=rgb, meta=dict(frame.meta)), metrics=metrics)


def stage_demosaic_stub(frame: Frame, params: Dict[str, Any]) -> StageResult:
    # Simple placeholder: replicate mosaic into 3 channels
    if frame.image.ndim == 2:
        rgb = np.repeat(frame.image[:, :, None], 3, axis=2).astype(np.float32)
    else:
        rgb = frame.image.astype(np.float32)
    return StageResult(frame=Frame(image=rgb, meta=dict(frame.meta)), metrics={"method": "replicate"})


def stage_oetf_encode_stub(frame: Frame, params: Dict[str, Any]) -> StageResult:
    # Pass-through; final encoding happens in runner
    return StageResult(frame=_copy_frame(frame), metrics={"encoding": "srgb", "bit_depth": 8})


def build_stage(name: str) -> Stage:
    mapping: Dict[str, Tuple[str, Callable[[Frame, Dict[str, Any]], StageResult]]] = {
        "raw_norm": ("RAW normalize", stage_raw_norm),
        "dpc": ("DPC", stage_dpc),
        "lsc": ("LSC", stage_lsc),
        "wb_gains": ("WB gains", stage_wb_gains),
        "demosaic": ("Demosaic", stage_demosaic),
        "denoise": ("Denoise", stage_stub_identity),
        "ccm": ("CCM", stage_stub_identity),
        "stats_3a": ("3A stats", stage_stub_identity),
        "tone": ("Tone", stage_stub_identity),
        "color_adjust": ("Color adjust", stage_stub_identity),
        "sharpen": ("Sharpen", stage_stub_identity),
        "oetf_encode": ("OETF encode", stage_oetf_encode_stub),
        "jdd_raw2rgb": ("JDD raw2rgb", stage_demosaic_stub),
        "drc_plus_color": ("DRC + color", stage_stub_identity),
    }
    if name not in mapping:
        raise ValueError(f"Unknown stage: {name}")
    display_name, func = mapping[name]
    return Stage(name=name, display_name=display_name, func=func)


def timed_call(func: Callable[..., StageResult], *args: Any, **kwargs: Any) -> Tuple[StageResult, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000.0
    return result, elapsed
