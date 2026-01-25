from __future__ import annotations

import json
from pathlib import Path

from mini_isp.pipeline import build_pipeline
from mini_isp.run import DEFAULT_CONFIG, run_pipeline, stage_dir_name


def test_end_to_end_smoke(tmp_path: Path) -> None:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["input"]["path"] = str(tmp_path / "sample.png")
    config["output"]["dir"] = str(tmp_path / "runs")
    config["output"]["name"] = "smoke"
    config["pipeline_mode"] = "classic"
    config["dump"]["enable"] = True
    config["dump"]["roi"]["enable"] = True

    run_root = Path(run_pipeline(config))
    manifest_path = run_root / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "0.1"
    assert manifest["pipeline_mode"] == "classic"
    assert len(manifest["stages"]) == len(build_pipeline("classic"))

    stages_root = run_root / "stages"
    for index, stage in enumerate(build_pipeline("classic")):
        stage_dir = stages_root / stage_dir_name(index, stage.name)
        assert (stage_dir / "preview.png").exists()
        assert (stage_dir / "debug.json").exists()
        assert (stage_dir / "timing_ms.json").exists()
        assert (stage_dir / "roi.png").exists()
