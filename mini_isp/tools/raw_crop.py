from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np

from mini_isp.io_utils import load_raw_mosaic, normalize_raw_mosaic


def _shift_cfa_pattern(cfa: str, x: int, y: int) -> str:
    cfa = (cfa or "RGGB").upper()
    if len(cfa) != 4:
        return cfa
    if (x % 2 == 0) and (y % 2 == 0):
        return cfa
    grid = np.array([[cfa[0], cfa[1]], [cfa[2], cfa[3]]], dtype=object)
    if y % 2 == 1:
        grid = np.roll(grid, -1, axis=0)
    if x % 2 == 1:
        grid = np.roll(grid, -1, axis=1)
    return "".join(grid.flatten().tolist())


def _validate_bounds(x: int, y: int, w: int, h: int, shape: Tuple[int, int]) -> None:
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("x/y must be >= 0 and w/h must be > 0")
    if y + h > shape[0] or x + w > shape[1]:
        raise ValueError("Crop is out of bounds for the input mosaic")


def _write_meta(path: str, meta: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def run_crop(
    input_path: str,
    out_dir: str,
    x: int,
    y: int,
    w: int,
    h: int,
    dtype: str,
    overwrite: bool,
) -> Tuple[str, str]:
    mosaic, meta = load_raw_mosaic(input_path, "RGGB")
    mosaic = np.asarray(mosaic)
    full_shape = (int(mosaic.shape[0]), int(mosaic.shape[1]))
    _validate_bounds(x, y, w, h, full_shape)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    crop_path = os.path.join(out_dir, "crop.npy")
    meta_path = os.path.join(out_dir, "meta.json")
    if not overwrite and (os.path.exists(crop_path) or os.path.exists(meta_path)):
        raise FileExistsError("Output exists; use --overwrite to replace")

    crop = mosaic[y : y + h, x : x + w]
    cfa_pattern = _shift_cfa_pattern(meta.get("cfa_pattern", "RGGB"), x, y)

    if dtype == "float32":
        black_level = float(meta.get("black_level", 0.0))
        white_level = meta.get("white_level")
        if white_level is None:
            if np.issubdtype(mosaic.dtype, np.integer):
                white_level = float(np.iinfo(mosaic.dtype).max)
            else:
                white_level = float(np.max(mosaic))
        crop_out = normalize_raw_mosaic(crop, black_level, float(white_level))
        dtype_note = "normalized_float32"
    elif dtype == "uint16":
        crop_out = crop.astype(np.uint16, copy=False)
        dtype_note = "uint16_raw"
        black_level = float(meta.get("black_level", 0.0))
        white_level = meta.get("white_level")
        if white_level is None:
            if np.issubdtype(mosaic.dtype, np.integer):
                white_level = float(np.iinfo(mosaic.dtype).max)
            else:
                white_level = float(np.max(mosaic))
    else:
        raise ValueError("dtype must be float32 or uint16")

    np.save(crop_path, crop_out)

    meta_out = {
        "source_path": input_path,
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "mosaic_shape": [full_shape[0], full_shape[1]],
        "crop_shape": [int(h), int(w)],
        "cfa_pattern": cfa_pattern,
        "black_level": float(black_level),
        "white_level": float(white_level),
        "bit_depth": meta.get("bit_depth"),
        "dtype": dtype_note,
    }
    _write_meta(meta_path, meta_out)
    return crop_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic RAW crops.")
    parser.add_argument("--input", required=True, type=str, help="Path to RAW/DNG input")
    parser.add_argument("--out", required=True, type=str, help="Output directory")
    parser.add_argument("--x", required=True, type=int, help="Crop x offset")
    parser.add_argument("--y", required=True, type=int, help="Crop y offset")
    parser.add_argument("--w", required=True, type=int, help="Crop width")
    parser.add_argument("--h", required=True, type=int, help="Crop height")
    parser.add_argument("--dtype", choices=["float32", "uint16"], default="float32")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    run_crop(
        input_path=args.input,
        out_dir=args.out,
        x=args.x,
        y=args.y,
        w=args.w,
        h=args.h,
        dtype=args.dtype,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
