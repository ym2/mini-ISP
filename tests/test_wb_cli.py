from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mini_isp.run import _resolve_wb_gains_stage_params


def test_resolve_wb_gains_explicit_config_wins() -> None:
    stage_cfg = {"wb_gains": [2.0, 3.0, 5.0]}
    frame_meta = {"input_kind": "raw", "wb_gains": [9.0, 9.0, 9.0], "wb_source": "camera_whitebalance"}
    out = _resolve_wb_gains_stage_params(
        stage_config=stage_cfg,
        frame_meta=frame_meta,
        cli_wb_mode="unity",
        cli_wb_gains=None,
    )
    # Explicit stage gains win; CLI intent should not override them.
    assert out["wb_gains"] == [2.0, 3.0, 5.0]
    assert out["wb_mode"] == "manual"
    assert out["wb_source"] == "manual"


def test_cli_wb_mode_manual_requires_wb_gains(tmp_path: Path) -> None:
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
        "wb_manual_missing",
        "--wb-mode",
        "manual",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "--wb-gains" in (result.stderr + result.stdout)


def test_cli_wb_mode_unity_records_debug(tmp_path: Path) -> None:
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
        "wb_unity",
        "--wb-mode",
        "unity",
    ]
    subprocess.check_call(cmd)
    debug_path = run_dir / "wb_unity" / "stages" / "03_wb_gains" / "debug.json"
    data = json.loads(debug_path.read_text(encoding="utf-8"))
    assert data["params"]["wb_mode"] == "unity"
    assert data["params"]["wb_source"] == "unity"
    assert data["params"]["wb_gains"] == [1.0, 1.0, 1.0]

