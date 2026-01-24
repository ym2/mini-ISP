from __future__ import annotations

from typing import List

from .stages import build_stage, Stage


PIPELINE_MODES = {
    "classic": [
        "raw_norm",
        "dpc",
        "lsc",
        "wb_gains",
        "demosaic",
        "denoise",
        "ccm",
        "stats_3a",
        "tone",
        "color_adjust",
        "sharpen",
        "oetf_encode",
    ],
    "jdd": [
        "raw_norm",
        "dpc",
        "lsc",
        "wb_gains",
        "jdd_raw2rgb",
        "ccm",
        "stats_3a",
        "tone",
        "color_adjust",
        "sharpen",
        "oetf_encode",
    ],
    "drc_plus": [
        "raw_norm",
        "dpc",
        "lsc",
        "wb_gains",
        "demosaic",
        "denoise",
        "ccm",
        "stats_3a",
        "drc_plus_color",
        "sharpen",
        "oetf_encode",
    ],
}


def build_pipeline(mode: str) -> List[Stage]:
    if mode not in PIPELINE_MODES:
        raise ValueError(f"Unknown pipeline_mode: {mode}")
    return [build_stage(name) for name in PIPELINE_MODES[mode]]
