from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mini_isp.config_overrides import apply_overrides


def test_apply_overrides_types_and_nested() -> None:
    base = {"stages": {"denoise": {"method": "gaussian"}}}
    overrides = [
        "stages.denoise.method=chroma_gaussian",
        "stages.denoise.sigma_y=1.0",
        "stages.denoise.ksize=5",
        "metrics.enable=true",
        "metrics.notes=null",
    ]
    out = apply_overrides(base, overrides)
    assert out["stages"]["denoise"]["method"] == "chroma_gaussian"
    assert out["stages"]["denoise"]["sigma_y"] == 1.0
    assert out["stages"]["denoise"]["ksize"] == 5
    assert out["metrics"]["enable"] is True
    assert out["metrics"]["notes"] is None


def test_apply_overrides_non_dict_path_error() -> None:
    base = {"stages": "oops"}
    with pytest.raises(ValueError):
        apply_overrides(base, ["stages.denoise.method=gaussian"])


def test_cli_set_overrides_denoise(tmp_path: Path) -> None:
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
        "cli_set",
        "--set",
        "stages.denoise.method=chroma_gaussian",
        "--set",
        "stages.denoise.sigma_y=1.0",
        "--set",
        "stages.denoise.sigma_c=2.0",
        "--set",
        "stages.denoise.ksize=5",
    ]
    subprocess.check_call(cmd)
    debug_path = run_dir / "cli_set" / "stages" / "05_denoise" / "debug.json"
    data = json.loads(debug_path.read_text(encoding="utf-8"))
    assert data["params"]["method"] == "chroma_gaussian"
