from __future__ import annotations

import json
from pathlib import Path

from mini_isp.pipeline import build_pipeline
from mini_isp.run import DEFAULT_CONFIG, run_pipeline, stage_dir_name


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_metrics_and_diagnostics_determinism(tmp_path: Path) -> None:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["input"]["path"] = str(tmp_path / "sample.png")
    config["output"]["dir"] = str(tmp_path / "runs")
    config["output"]["name"] = "metrics_a"
    config["dump"]["enable"] = True
    config["metrics"]["enable"] = True
    config["metrics"]["diagnostics"] = True

    run_a = Path(run_pipeline(config))
    config["output"]["name"] = "metrics_b"
    run_b = Path(run_pipeline(config))

    pipeline = build_pipeline("classic")
    for index, stage in enumerate(pipeline):
        stage_dir_a = run_a / "stages" / stage_dir_name(index, stage.name) / "extra"
        stage_dir_b = run_b / "stages" / stage_dir_name(index, stage.name) / "extra"
        metrics_a = stage_dir_a / "metrics.json"
        metrics_b = stage_dir_b / "metrics.json"
        assert metrics_a.exists()
        assert metrics_b.exists()
        assert _load_text(metrics_a) == _load_text(metrics_b)

        if index > 0:
            diff_a = stage_dir_a / "diff_metrics.json"
            diff_b = stage_dir_b / "diff_metrics.json"
            assert diff_a.exists()
            assert diff_b.exists()
            assert _load_text(diff_a) == _load_text(diff_b)

        diag_a = stage_dir_a / "diagnostics"
        diag_b = stage_dir_b / "diagnostics"
        for name in ["false_color.png", "zipper.png", "halo.png"]:
            file_a = diag_a / name
            file_b = diag_b / name
            assert file_a.exists()
            assert file_b.exists()
            assert file_a.read_bytes() == file_b.read_bytes()


def test_cli_overrides_metrics_and_diagnostics(tmp_path: Path) -> None:
    import subprocess
    import sys

    run_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "-m",
        "mini_isp.run",
        "--input",
        str(tmp_path / "sample.png"),
        "--out",
        str(run_dir),
        "--pipeline_mode",
        "classic",
        "--name",
        "cli_metrics",
        "--enable-metrics",
        "--enable-diagnostics",
        "--metrics-target",
        "preview",
        "--metrics-diff",
        "l1",
        "--metrics-out",
        "extra",
    ]
    subprocess.check_call(cmd)
    stage0 = run_dir / "cli_metrics" / "stages" / "00_raw_norm" / "extra"
    assert (stage0 / "metrics.json").exists()
    assert (stage0 / "diagnostics" / "false_color.png").exists()


def test_cli_metrics_target_linear_fails(tmp_path: Path) -> None:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "mini_isp.run",
        "--input",
        str(tmp_path / "sample.png"),
        "--out",
        str(tmp_path / "runs"),
        "--pipeline_mode",
        "classic",
        "--name",
        "cli_metrics",
        "--enable-metrics",
        "--metrics-target",
        "linear",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "linear metrics not implemented yet" in (result.stderr + result.stdout)
