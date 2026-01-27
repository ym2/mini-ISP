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
        "clip_applied": clip_applied,
        "clip_range": clip_range,
        "min": float(np.min(rgb)),
        "max": float(np.max(rgb)),
        "p01": float(np.percentile(rgb, 1.0)),
        "p99": float(np.percentile(rgb, 99.0)),
        "resolved_params": {"method": method, "clip": bool(params.get("clip", False))},
    }
    return StageResult(frame=Frame(image=rgb, meta=dict(frame.meta)), metrics=metrics)


def _kernel_box(ksize: int) -> np.ndarray:
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("ksize must be odd and >= 1")
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    return kernel / float(ksize * ksize)


def _kernel_gaussian(ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("ksize must be odd and >= 1")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    radius = ksize // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(x ** 2) / (2 * sigma * sigma))
    g /= np.sum(g)
    kernel = np.outer(g, g).astype(np.float32)
    return kernel / np.sum(kernel)


def _convolve2d_edge(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(image, dtype=np.float32)
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            patch = padded[y : y + kh, x : x + kw]
            out[y, x] = float(np.sum(patch * kernel))
    return out.astype(np.float32)


def stage_denoise(frame: Frame, params: Dict[str, Any]) -> StageResult:
    image = frame.image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        return StageResult(frame=_copy_frame(frame), metrics={"warning": "denoise expects RGB image"})
    method = str(params.get("method", "gaussian")).lower()
    clip_applied = False
    clip_range = None
    edge_mode = "edge"
    if method == "gaussian":
        ksize = int(params.get("ksize", 5))
        sigma = float(params.get("sigma", 1.0))
        kernel = _kernel_gaussian(ksize, sigma)
        method_params = {"ksize": ksize, "sigma": sigma}
    elif method == "box":
        ksize = int(params.get("ksize", 3))
        kernel = _kernel_box(ksize)
        method_params = {"ksize": ksize}
    elif method == "chroma_gaussian":
        ksize = int(params.get("ksize", 5))
        sigma_y = float(params.get("sigma_y", 1.0))
        sigma_c = float(params.get("sigma_c", 2.0))
        kernel_y = _kernel_gaussian(ksize, sigma_y)
        kernel_c = _kernel_gaussian(ksize, sigma_c)
        method_params = {"ksize": ksize, "sigma_y": sigma_y, "sigma_c": sigma_c}
        y = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
        c1 = image[:, :, 0] - image[:, :, 1]
        c2 = image[:, :, 2] - image[:, :, 1]
        y_f = _convolve2d_edge(y, kernel_y)
        c1_f = _convolve2d_edge(c1, kernel_c)
        c2_f = _convolve2d_edge(c2, kernel_c)
        g = y_f - 0.2126 * c1_f - 0.0722 * c2_f
        r = g + c1_f
        b = g + c2_f
        out = np.stack([r, g, b], axis=2).astype(np.float32)
    else:
        raise ValueError(f"Unknown denoise method: {method}")

    if method in ("gaussian", "box"):
        channels = []
        for c in range(3):
            channels.append(_convolve2d_edge(image[:, :, c], kernel))
        out = np.stack(channels, axis=2).astype(np.float32)

    if params.get("clip", False):
        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        clip_applied = True
        clip_range = [0.0, 1.0]

    metrics = {
        "derived": {"edge_mode": edge_mode},
        "clip_applied": clip_applied,
        "clip_range": clip_range,
        "min": float(np.min(out)),
        "max": float(np.max(out)),
        "p01": float(np.percentile(out, 1.0)),
        "p99": float(np.percentile(out, 99.0)),
        "resolved_params": {"method": method, **method_params, "edge_mode": edge_mode},
    }
    return StageResult(frame=Frame(image=out, meta=dict(frame.meta)), metrics=metrics)


def _apply_ccm(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    flat = image.reshape(-1, 3)
    out = flat @ matrix.T
    return out.reshape(h, w, 3).astype(np.float32)


def stage_ccm(frame: Frame, params: Dict[str, Any]) -> StageResult:
    image = frame.image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        return StageResult(frame=_copy_frame(frame), metrics={"warning": "ccm expects RGB image"})

    mode = str(params.get("mode", "identity")).lower()
    if mode == "identity":
        matrix = np.eye(3, dtype=np.float32)
    elif mode == "manual":
        matrix = np.array(params.get("matrix", np.eye(3)), dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError("ccm manual mode requires a 3x3 matrix")
    elif mode == "profile":
        # Stub: fall back to identity
        matrix = np.eye(3, dtype=np.float32)
    else:
        raise ValueError(f"Unknown ccm mode: {mode}")

    out = _apply_ccm(image, matrix)

    clip_applied = False
    clip_range = None
    clip = params.get("clip", None)
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        out = np.clip(out, lo, hi).astype(np.float32)
        clip_applied = True
        clip_range = [lo, hi]

    metrics = {
        "clip_applied": clip_applied,
        "clip_range": clip_range,
        "min": float(np.min(out)),
        "max": float(np.max(out)),
        "p01": float(np.percentile(out, 1.0)),
        "p99": float(np.percentile(out, 99.0)),
        "resolved_params": {"mode": mode, "matrix": matrix.tolist(), "clip": clip},
    }
    if mode == "profile":
        metrics["warnings"] = ["profile mode not implemented; used identity"]

    meta = dict(frame.meta)
    meta["ccm"] = matrix.tolist()
    meta["ccm_mode"] = mode
    return StageResult(frame=Frame(image=out, meta=meta), metrics=metrics)


def _sobel_edges(luma: np.ndarray) -> np.ndarray:
    # Edge-clamped Sobel gradients
    padded = np.pad(luma, 1, mode="edge")
    gx = (
        -1 * padded[:-2, :-2]
        + 1 * padded[:-2, 2:]
        - 2 * padded[1:-1, :-2]
        + 2 * padded[1:-1, 2:]
        - 1 * padded[2:, :-2]
        + 1 * padded[2:, 2:]
    )
    gy = (
        -1 * padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - 1 * padded[:-2, 2:]
        + 1 * padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + 1 * padded[2:, 2:]
    )
    return gx, gy


def _center_crop(image: np.ndarray, frac: float = 0.5) -> np.ndarray:
    h, w = image.shape[:2]
    ch = int(round(h * frac))
    cw = int(round(w * frac))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return image[y0 : y0 + ch, x0 : x0 + cw]


def stage_stats_3a(frame: Frame, params: Dict[str, Any]) -> StageResult:
    image = frame.image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        return StageResult(frame=_copy_frame(frame), metrics={"warning": "stats_3a expects RGB image"})

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # AE stats
    luma_clipped = np.clip(luma, 0.0, 1.0)
    hist, bin_edges = np.histogram(luma_clipped, bins=64, range=(0.0, 1.0))
    clip_pct = float(np.mean((luma <= 0.0) | (luma >= 1.0)) * 100.0)
    ae_stats = {
        "mean": float(np.mean(luma)),
        "p01": float(np.percentile(luma, 1.0)),
        "p99": float(np.percentile(luma, 99.0)),
        "clip_pct": clip_pct,
        "hist": hist.tolist(),
        "hist_bins": bin_edges.tolist(),
    }

    # AWB stats
    awb_stats = {
        "mean_r": float(np.mean(r)),
        "mean_g": float(np.mean(g)),
        "mean_b": float(np.mean(b)),
    }

    # AF stats (Tenengrad on ROI or center crop)
    roi_cfg = params.get("roi", {})
    if roi_cfg.get("enable", False):
        xywh = roi_cfg.get("xywh", [0.35, 0.35, 0.30, 0.30])
        x, y, w, h = xywh
        hh, ww = luma.shape
        x0 = int(round(np.clip(x, 0.0, 1.0) * ww))
        y0 = int(round(np.clip(y, 0.0, 1.0) * hh))
        x1 = int(round(np.clip(x + w, 0.0, 1.0) * ww))
        y1 = int(round(np.clip(y + h, 0.0, 1.0) * hh))
        x1 = max(x1, x0 + 1)
        y1 = max(y1, y0 + 1)
        roi = luma[y0:y1, x0:x1]
        roi_used = "config"
    else:
        roi = _center_crop(luma, 0.5)
        roi_used = "center_crop"
    gx, gy = _sobel_edges(roi)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    af_stats = {"tenengrad": tenengrad, "roi": roi_used}

    stats_3a = {"ae": ae_stats, "awb": awb_stats, "af": af_stats}
    metrics = {"stats_3a": stats_3a}

    meta = dict(frame.meta)
    meta["stats_3a"] = stats_3a
    return StageResult(frame=Frame(image=np.array(frame.image, copy=True), meta=meta), metrics=metrics)
def stage_jdd_raw2rgb(frame: Frame, params: Dict[str, Any]) -> StageResult:
    method = str(params.get("method", "wrapper")).lower()
    if method != "wrapper":
        raise ValueError(f"Unknown jdd_raw2rgb method: {method}")
    demosaic_params = params.get("demosaic", {})
    denoise_params = params.get("denoise", {})
    demosaic_result = stage_demosaic(frame, demosaic_params)
    denoise_result = stage_denoise(demosaic_result.frame, denoise_params)
    metrics = {
        "method": method,
        "demosaic": demosaic_result.metrics,
        "denoise": denoise_result.metrics,
    }
    return StageResult(frame=denoise_result.frame, metrics=metrics)


def stage_demosaic_stub(frame: Frame, params: Dict[str, Any]) -> StageResult:
    # Simple placeholder: replicate mosaic into 3 channels
    if frame.image.ndim == 2:
        rgb = np.repeat(frame.image[:, :, None], 3, axis=2).astype(np.float32)
    else:
        rgb = frame.image.astype(np.float32)
    return StageResult(frame=Frame(image=rgb, meta=dict(frame.meta)), metrics={"method": "replicate"})


def _tone_reinhard(image: np.ndarray) -> np.ndarray:
    return image / (1.0 + image)


def _tone_filmic(image: np.ndarray) -> np.ndarray:
    # Simple filmic-like curve (Hable-ish)
    a = 0.22
    b = 0.30
    c = 0.10
    d = 0.20
    e = 0.01
    f = 0.30
    x = image
    out = ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f
    return out


def stage_tone(frame: Frame, params: Dict[str, Any]) -> StageResult:
    image = frame.image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        return StageResult(frame=_copy_frame(frame), metrics={"warning": "tone expects RGB image"})
    method = str(params.get("method", "reinhard")).lower()
    if method == "reinhard":
        out = _tone_reinhard(image)
    elif method == "filmic":
        out = _tone_filmic(image)
    else:
        raise ValueError(f"Unknown tone method: {method}")

    clip_applied = False
    clip_range = None
    clip = params.get("clip", None)
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        out = np.clip(out, lo, hi).astype(np.float32)
        clip_applied = True
        clip_range = [lo, hi]

    metrics = {
        "clip_applied": clip_applied,
        "clip_range": clip_range,
        "min": float(np.min(out)),
        "max": float(np.max(out)),
        "p01": float(np.percentile(out, 1.0)),
        "p99": float(np.percentile(out, 99.0)),
        "resolved_params": {"method": method, "clip": clip},
    }
    return StageResult(frame=Frame(image=out.astype(np.float32), meta=dict(frame.meta)), metrics=metrics)


def stage_color_adjust(frame: Frame, params: Dict[str, Any]) -> StageResult:
    image = frame.image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        return StageResult(frame=_copy_frame(frame), metrics={"warning": "color_adjust expects RGB image"})
    method = str(params.get("method", "identity")).lower()
    if method == "identity":
        out = np.array(image, copy=True)
        sat_scale = 1.0
    elif method == "chroma_scale_lrgb":
        sat_scale = float(params.get("sat_scale", 1.0))
        y = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
        y = y[:, :, None]
        out = y + sat_scale * (image - y)
    else:
        raise ValueError(f"Unknown color_adjust method: {method}")

    metrics = {
        "min": float(np.min(out)),
        "max": float(np.max(out)),
        "p01": float(np.percentile(out, 1.0)),
        "p99": float(np.percentile(out, 99.0)),
        "resolved_params": {"method": method, "sat_scale": sat_scale},
    }
    return StageResult(frame=Frame(image=out.astype(np.float32), meta=dict(frame.meta)), metrics=metrics)


def stage_sharpen(frame: Frame, params: Dict[str, Any]) -> StageResult:
    image = frame.image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        return StageResult(frame=_copy_frame(frame), metrics={"warning": "sharpen expects RGB image"})
    method = str(params.get("method", "unsharp_mask")).lower()
    sigma = float(params.get("sigma", 1.0))
    amount = float(params.get("amount", 0.5))
    threshold = float(params.get("threshold", 0.0))
    luma_only = bool(params.get("luma_only", True))
    gate_mode = str(params.get("gate_mode", "soft")).lower()
    edge_mode = "edge"

    ksize = int(max(3, int(round(sigma * 6 + 1))))
    if ksize % 2 == 0:
        ksize += 1
    kernel = _kernel_gaussian(ksize, sigma)

    if method == "unsharp_mask":
        blurred = np.stack(
            [_convolve2d_edge(image[:, :, c], kernel) for c in range(3)], axis=2
        ).astype(np.float32)
        mask = image - blurred
        if threshold > 0.0:
            mask = np.where(np.abs(mask) >= threshold, mask, 0.0)
        out = image + amount * mask
    elif method == "unsharp_mask_tuned":
        if gate_mode != "soft":
            raise ValueError(f"Unknown gate_mode: {gate_mode}")
        if luma_only:
            y = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
            blurred_y = _convolve2d_edge(y, kernel).astype(np.float32)
            mask = (y - blurred_y).astype(np.float32)
            eps = 1e-6
            denom = max(threshold, eps)
            gain = np.clip((np.abs(mask) - threshold) / denom, 0.0, 1.0)
            gated = mask * gain
            out = image + amount * gated[:, :, None]
        else:
            blurred = np.stack(
                [_convolve2d_edge(image[:, :, c], kernel) for c in range(3)], axis=2
            ).astype(np.float32)
            mask = image - blurred
            eps = 1e-6
            denom = max(threshold, eps)
            gain = np.clip((np.abs(mask) - threshold) / denom, 0.0, 1.0)
            gated = mask * gain
            out = image + amount * gated
    else:
        raise ValueError(f"Unknown sharpen method: {method}")

    metrics = {
        "method": method,
        "params": {
            "sigma": sigma,
            "amount": amount,
            "threshold": threshold,
            "luma_only": luma_only,
            "gate_mode": gate_mode,
            "edge_mode": edge_mode,
        },
        "clip_applied": False,
        "clip_range": None,
        "min": float(np.min(out)),
        "max": float(np.max(out)),
        "p01": float(np.percentile(out, 1.0)),
        "p99": float(np.percentile(out, 99.0)),
        "resolved_params": {
            "method": method,
            "sigma": sigma,
            "amount": amount,
            "threshold": threshold,
            "luma_only": luma_only,
            "gate_mode": gate_mode,
            "edge_mode": edge_mode,
        },
    }
    return StageResult(frame=Frame(image=out.astype(np.float32), meta=dict(frame.meta)), metrics=metrics)


def stage_oetf_encode(frame: Frame, params: Dict[str, Any]) -> StageResult:
    image = frame.image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        return StageResult(frame=_copy_frame(frame), metrics={"warning": "oetf_encode expects RGB image"})
    oetf = str(params.get("oetf", "srgb")).lower()
    if oetf != "srgb":
        raise ValueError(f"Unknown oetf: {oetf}")
    # Required clipping before encoding
    clipped = np.clip(image, 0.0, 1.0).astype(np.float32)
    encoded = np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1 / 2.4) - 0.055,
    )
    encoded_u8 = np.clip(encoded * 255.0 + 0.5, 0, 255).astype(np.uint8)

    metrics = {
        "clip_applied": True,
        "clip_range": [0.0, 1.0],
        "dtype": "uint8",
        "bit_depth": 8,
        "resolved_params": {"oetf": oetf},
    }
    return StageResult(frame=Frame(image=encoded_u8, meta=dict(frame.meta)), metrics=metrics)


def stage_drc_plus_color(frame: Frame, params: Dict[str, Any]) -> StageResult:
    method = str(params.get("method", "wrapper")).lower()
    if method != "wrapper":
        raise ValueError(f"Unknown drc_plus_color method: {method}")

    tone_params = params.get("tone", {})
    color_params = params.get("color_adjust", {})

    tone_result = stage_tone(frame, tone_params)
    color_result = stage_color_adjust(tone_result.frame, color_params)

    metrics = {
        "method": method,
        "expands_to": ["tone", "color_adjust"],
        "tone": {
            "method": tone_result.metrics.get("method"),
            "params": tone_params,
        },
        "color_adjust": {
            "method": color_result.metrics.get("method"),
            "params": color_params,
        },
    }
    return StageResult(frame=color_result.frame, metrics=metrics)

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
        "denoise": ("Denoise", stage_denoise),
        "ccm": ("CCM", stage_ccm),
        "stats_3a": ("3A stats", stage_stats_3a),
        "tone": ("Tone", stage_tone),
        "color_adjust": ("Color adjust", stage_color_adjust),
        "sharpen": ("Sharpen", stage_sharpen),
        "oetf_encode": ("OETF encode", stage_oetf_encode),
        "jdd_raw2rgb": ("JDD raw2rgb", stage_jdd_raw2rgb),
        "drc_plus_color": ("DRC + color", stage_drc_plus_color),
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
