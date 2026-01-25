from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mini_isp.io_utils import Frame
from mini_isp.pipeline import build_pipeline
from mini_isp.run import DEFAULT_CONFIG, run_pipeline, stage_dir_name
from mini_isp.stages import stage_color_adjust, stage_drc_plus_color, stage_tone


def test_drc_plus_matches_tone_then_color_adjust() -> None:
    img = np.random.default_rng(0).random((4, 4, 3), dtype=np.float32)
    frame = Frame(image=img, meta={})
    tone_params = {"method": "reinhard"}
    color_params = {"method": "chroma_scale_lrgb", "sat_scale": 1.2}
    seq = stage_color_adjust(stage_tone(frame, tone_params).frame, color_params).frame.image
    composite = stage_drc_plus_color(
        frame, {"method": "wrapper", "tone": tone_params, "color_adjust": color_params}
    ).frame.image
    assert np.allclose(seq, composite, atol=1e-6)


def test_drc_plus_pipeline_artifacts(tmp_path: Path) -> None:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["input"]["path"] = str(tmp_path / "sample.png")
    config["output"]["dir"] = str(tmp_path / "runs")
    config["output"]["name"] = "drc_plus_smoke"
    config["pipeline_mode"] = "drc_plus"
    config["dump"]["enable"] = True
    config["dump"]["roi"]["enable"] = True

    run_root = Path(run_pipeline(config))
    stages_root = run_root / "stages"
    pipeline = build_pipeline("drc_plus")
    for index, stage in enumerate(pipeline):
        stage_dir = stages_root / stage_dir_name(index, stage.name)
        assert (stage_dir / "preview.png").exists()
        assert (stage_dir / "debug.json").exists()
        assert (stage_dir / "timing_ms.json").exists()
        assert (stage_dir / "roi.png").exists()
