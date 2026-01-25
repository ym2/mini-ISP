from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .io_utils import downscale_max_side, roi_crop, save_png, to_display_u8, write_json


def build_preview_image(image: np.ndarray) -> np.ndarray:
    return to_display_u8(image)


def write_stage_artifacts(
    stage_root: str,
    stage_name: str,
    image: np.ndarray,
    stage_params: Dict[str, Any],
    metrics: Dict[str, Any],
    timing_ms: float,
    dump_config: Dict[str, Any],
) -> None:
    preview = build_preview_image(image)
    preview = downscale_max_side(preview, dump_config["preview_max_side"])
    save_png(f"{stage_root}/preview.png", preview)

    if dump_config["roi"]["enable"]:
        roi = roi_crop(preview, tuple(dump_config["roi"]["xywh"]))
        save_png(f"{stage_root}/roi.png", roi)

    debug = {
        "stage": stage_name,
        "params": stage_params,
        "metrics": metrics,
        "warnings": [],
        "notes": [],
    }
    write_json(f"{stage_root}/debug.json", debug)
    write_json(f"{stage_root}/timing_ms.json", {"timing_ms": timing_ms})
