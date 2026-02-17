from __future__ import annotations

import json
import os
import subprocess
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


_D50_WHITE_XYZ = np.array([0.96422, 1.0, 0.82521], dtype=np.float32)
_XYZ_TO_LIN_SRGB_D65 = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float32,
)


def _read_exiftool_metadata(path: str) -> Tuple[Optional[Dict[str, Any]], str]:
    cmd = ["exiftool", "-j", "-n", "-s", "-u", "-a", path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None, "exiftool_not_found"
    except Exception:
        return None, "exiftool_exec_error"

    if proc.returncode != 0:
        return None, "exiftool_failed"
    try:
        parsed = json.loads(proc.stdout)
    except Exception:
        return None, "exiftool_invalid_json"
    if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
        return None, "exiftool_empty_metadata"
    return parsed[0], "ok"


def _exiftool_get(meta: Dict[str, Any], tag: str) -> Optional[Any]:
    if tag in meta:
        return meta[tag]
    suffix = f":{tag}"
    for key, value in meta.items():
        if key.endswith(suffix):
            return value
    return None


def _parse_exiftool_mat3(meta: Dict[str, Any], tag: str) -> Optional[np.ndarray]:
    value = _exiftool_get(meta, tag)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [float(x) for x in value]
    else:
        parts = [float(x) for x in str(value).split()]
    if len(parts) != 9:
        return None
    mat = np.asarray(parts, dtype=np.float32).reshape(3, 3)
    if not np.all(np.isfinite(mat)):
        return None
    return mat


def _parse_exiftool_vec3(meta: Dict[str, Any], tag: str) -> Optional[np.ndarray]:
    value = _exiftool_get(meta, tag)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [float(x) for x in value]
    else:
        parts = [float(x) for x in str(value).split()]
    if len(parts) != 3:
        return None
    vec = np.asarray(parts, dtype=np.float32)
    if not np.all(np.isfinite(vec)):
        return None
    return vec


def _illuminant_code_to_cct(code: Any) -> Optional[float]:
    if code is None:
        return None
    try:
        c = int(code)
    except Exception:
        return None
    table = {
        1: 5500.0,
        2: 4200.0,
        3: 2850.0,
        4: 5500.0,
        9: 5500.0,
        10: 6500.0,
        11: 7500.0,
        12: 6400.0,
        13: 5000.0,
        14: 4000.0,
        15: 3200.0,
        17: 2856.0,
        18: 4874.0,
        19: 6774.0,
        20: 5503.0,
        21: 6504.0,
        22: 7504.0,
        23: 5003.0,
        24: 3200.0,
    }
    return table.get(c)


def _cct_from_xyz_mccamy(xyz: np.ndarray) -> Optional[float]:
    v = np.asarray(xyz, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(v)):
        return None
    s = float(np.sum(v))
    if s <= 1e-12:
        return None
    x = float(v[0] / s)
    y = float(v[1] / s)
    denom = 0.1858 - y
    if abs(denom) < 1e-12:
        return None
    n = (x - 0.3320) / denom
    cct = -449.0 * (n**3) + 3525.0 * (n**2) - 6823.3 * n + 5520.33
    if not np.isfinite(cct) or cct <= 0.0:
        return None
    return float(cct)


def _metadata_interp_weight(
    *,
    as_shot_neutral: Optional[np.ndarray],
    color_matrix_1: Optional[np.ndarray],
    color_matrix_2: Optional[np.ndarray],
    calibration_illuminant_1: Any,
    calibration_illuminant_2: Any,
) -> Tuple[float, str]:
    t1 = _illuminant_code_to_cct(calibration_illuminant_1)
    t2 = _illuminant_code_to_cct(calibration_illuminant_2)
    if as_shot_neutral is None or color_matrix_1 is None or color_matrix_2 is None:
        return 0.5, "fallback_0p5"
    if t1 is None or t2 is None:
        return 0.5, "fallback_0p5"
    try:
        cm_mid = 0.5 * np.asarray(color_matrix_1, dtype=np.float64) + 0.5 * np.asarray(color_matrix_2, dtype=np.float64)
        inv_mid = np.linalg.inv(cm_mid)
        xyz_scene = inv_mid @ np.asarray(as_shot_neutral, dtype=np.float64).reshape(3)
    except Exception:
        return 0.5, "fallback_0p5"
    cct_scene = _cct_from_xyz_mccamy(xyz_scene)
    if cct_scene is None:
        return 0.5, "fallback_0p5"
    denom = (1.0 / t1) - (1.0 / t2)
    if abs(denom) < 1e-12:
        return 0.5, "fallback_0p5"
    w_raw = ((1.0 / cct_scene) - (1.0 / t2)) / denom
    return float(np.clip(w_raw, 0.0, 1.0)), "metadata_cct"


def _interp_mat3(m1: Optional[np.ndarray], m2: Optional[np.ndarray], w: float) -> Optional[np.ndarray]:
    if m1 is None and m2 is None:
        return None
    if m1 is None:
        return m2
    if m2 is None:
        return m1
    return (w * m1 + (1.0 - w) * m2).astype(np.float32)


def _synthesize_native_fm(
    analog_balance: Optional[np.ndarray],
    camera_calibration: Optional[np.ndarray],
    color_matrix: np.ndarray,
) -> np.ndarray:
    cm = np.asarray(color_matrix, dtype=np.float64)
    if camera_calibration is not None:
        cc = np.asarray(camera_calibration, dtype=np.float64)
    else:
        cc = np.eye(3, dtype=np.float64)
    if analog_balance is not None:
        ab = np.diag(np.asarray(analog_balance, dtype=np.float64))
    else:
        ab = np.eye(3, dtype=np.float64)
    m_native = ab @ cc @ cm
    cam_neutral = m_native @ _D50_WHITE_XYZ
    fm = np.linalg.inv(m_native) @ np.diag(cam_neutral)
    return fm.astype(np.float32)


def _xyz_whitepoint_d65() -> np.ndarray:
    return np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def _bradford_adaptation_matrix(src_white_xyz: np.ndarray, dst_white_xyz: np.ndarray) -> np.ndarray:
    src = np.asarray(src_white_xyz, dtype=np.float64).reshape(3)
    dst = np.asarray(dst_white_xyz, dtype=np.float64).reshape(3)
    b = np.array(
        [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ],
        dtype=np.float64,
    )
    b_inv = np.linalg.inv(b)
    src_lms = b @ src
    dst_lms = b @ dst
    scale = np.diag(dst_lms / np.maximum(src_lms, 1e-12))
    m = b_inv @ scale @ b
    return m.astype(np.float32)


def _xyz_d50_to_lin_srgb_d65() -> np.ndarray:
    m_xyz_d65_by_d50 = _bradford_adaptation_matrix(_D50_WHITE_XYZ, _xyz_whitepoint_d65())
    return (_XYZ_TO_LIN_SRGB_D65 @ m_xyz_d65_by_d50).astype(np.float32)


def derive_dng_ccm_from_exif_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    color_matrix_1 = _parse_exiftool_mat3(meta, "ColorMatrix1")
    color_matrix_2 = _parse_exiftool_mat3(meta, "ColorMatrix2")
    forward_matrix_1 = _parse_exiftool_mat3(meta, "ForwardMatrix1")
    forward_matrix_2 = _parse_exiftool_mat3(meta, "ForwardMatrix2")
    camera_calibration_1 = _parse_exiftool_mat3(meta, "CameraCalibration1")
    camera_calibration_2 = _parse_exiftool_mat3(meta, "CameraCalibration2")
    as_shot_neutral = _parse_exiftool_vec3(meta, "AsShotNeutral")
    analog_balance = _parse_exiftool_vec3(meta, "AnalogBalance")
    cal_ill_1 = _exiftool_get(meta, "CalibrationIlluminant1")
    cal_ill_2 = _exiftool_get(meta, "CalibrationIlluminant2")

    weight, weight_source = _metadata_interp_weight(
        as_shot_neutral=as_shot_neutral,
        color_matrix_1=color_matrix_1,
        color_matrix_2=color_matrix_2,
        calibration_illuminant_1=cal_ill_1,
        calibration_illuminant_2=cal_ill_2,
    )

    matrix_xyz_by_cam: Optional[np.ndarray] = None
    cam_source = "none"
    if forward_matrix_1 is not None or forward_matrix_2 is not None:
        matrix_xyz_by_cam = _interp_mat3(forward_matrix_1, forward_matrix_2, weight)
        cam_source = "dng_tags_forward_matrix"
    elif color_matrix_1 is not None or color_matrix_2 is not None:
        try:
            fm_1 = _synthesize_native_fm(analog_balance, camera_calibration_1, color_matrix_1) if color_matrix_1 is not None else None
            fm_2 = _synthesize_native_fm(analog_balance, camera_calibration_2, color_matrix_2) if color_matrix_2 is not None else None
            matrix_xyz_by_cam = _interp_mat3(fm_1, fm_2, weight)
            cam_source = "dng_tags_synthesized_native_chain_fm"
        except np.linalg.LinAlgError:
            return {"available": False, "reason": "dng_native_chain_inversion_failed"}
    else:
        return {"available": False, "reason": "no_usable_dng_matrices"}

    if matrix_xyz_by_cam is None:
        return {"available": False, "reason": "missing_interpolated_matrix"}
    if not np.all(np.isfinite(matrix_xyz_by_cam)) or float(np.linalg.norm(matrix_xyz_by_cam)) < 1e-8:
        return {"available": False, "reason": "invalid_dng_matrix"}

    xyz_to_working = _xyz_d50_to_lin_srgb_d65()
    return {
        "available": True,
        "cam_to_xyz_matrix": matrix_xyz_by_cam.astype(np.float32).tolist(),
        "cam_to_xyz_source": cam_source,
        "cam_to_xyz_space": "xyz_d50",
        "xyz_to_working_matrix": xyz_to_working.tolist(),
        "xyz_to_working_source": "constant_xyz_d50_to_lin_srgb_d65",
        "interp_weight_matrix1": float(weight),
        "interp_weight_source": weight_source,
    }


def derive_dng_ccm_from_file(path: str) -> Dict[str, Any]:
    meta, status = _read_exiftool_metadata(path)
    if meta is None:
        return {"available": False, "reason": status}
    return derive_dng_ccm_from_exif_metadata(meta)


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
        if mosaic.ndim != 2:
            raise ValueError(
                "RAW input is not a 2D Bayer mosaic "
                f"(got shape {tuple(int(x) for x in mosaic.shape)}). "
                "This file is likely a non-Bayer/RGB DNG and is unsupported by the RAW mosaic pipeline."
            )

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

    ccm_info: Dict[str, Any] = {"available": False, "reason": "non_dng_input"}
    if os.path.splitext(path)[1].lower() == ".dng":
        ccm_info = derive_dng_ccm_from_file(path)

    meta = {
        "cfa_pattern": cfa_pattern,
        "black_level": float(black_level) if black_level is not None else 0.0,
        "white_level": float(white_level) if white_level is not None else None,
        "bit_depth": bit_depth,
        "wb_gains": [float(x) for x in wb_gains],
        "wb_source": wb_source,
        "cam_to_xyz_source": "none",
        "xyz_to_working_source": "none",
        "ccm_auto_reason": ccm_info.get("reason"),
    }
    if ccm_info.get("available"):
        meta["cam_to_xyz_matrix"] = ccm_info["cam_to_xyz_matrix"]
        meta["cam_to_xyz_source"] = ccm_info.get("cam_to_xyz_source", "none")
        meta["xyz_to_working_matrix"] = ccm_info["xyz_to_working_matrix"]
        meta["xyz_to_working_source"] = ccm_info.get("xyz_to_working_source", "none")
        meta["ccm_auto_reason"] = "dng_tags_available"
    return mosaic, meta


def normalize_raw_mosaic(raw: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    eps = 1e-6
    denom = max(float(white_level) - float(black_level), eps)
    norm = (raw.astype(np.float32) - float(black_level)) / denom
    return np.clip(norm, 0.0, 1.0).astype(np.float32)
