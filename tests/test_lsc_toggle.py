from __future__ import annotations

import json
from pathlib import Path

from mini_isp.run import DEFAULT_CONFIG, run_pipeline, stage_dir_name


def test_lsc_enabled_false_skips_stage_and_keeps_preview_identical(tmp_path: Path) -> None:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["input"]["path"] = str(tmp_path / "sample.png")
    config["output"]["dir"] = str(tmp_path / "runs")
    config["output"]["name"] = "lsc_disabled"
    config["pipeline_mode"] = "classic"
    config["dump"]["enable"] = True
    config.setdefault("stages", {})
    config["stages"]["lsc"] = {"enabled": False}

    run_root = Path(run_pipeline(config))
    dpc_preview = run_root / "stages" / stage_dir_name(1, "dpc") / "preview.png"
    lsc_preview = run_root / "stages" / stage_dir_name(2, "lsc") / "preview.png"
    assert lsc_preview.read_bytes() == dpc_preview.read_bytes()

    debug = json.loads(
        (run_root / "stages" / stage_dir_name(2, "lsc") / "debug.json").read_text(encoding="utf-8")
    )
    assert debug["params"]["enabled"] is False
    assert debug["metrics"]["skipped"] is True

