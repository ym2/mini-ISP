from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _write_png(path: Path, value: int) -> None:
    arr = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_scene_pack_report(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    _write_png(inputs_dir / "a.png", 10)
    _write_png(inputs_dir / "b.png", 20)
    crop_dir = inputs_dir / "crop1"
    crop_dir.mkdir()
    np.save(crop_dir / "crop.npy", np.zeros((4, 4), dtype=np.float32))
    (crop_dir / "meta.json").write_text(
        json.dumps(
            {
                "cfa_pattern": "RGGB",
                "black_level": 0.0,
                "white_level": 1.0,
                "bit_depth": 10,
                "x": 0,
                "y": 0,
                "w": 4,
                "h": 4,
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "-m",
        "mini_isp.tools.scene_pack",
        "--inputs",
        str(inputs_dir),
        "--out",
        str(out_dir),
        "--name",
        "demo",
        "--baseline-set",
        "stages.denoise.method=gaussian",
        "--candidate-set",
        "stages.denoise.method=chroma_gaussian",
    ]
    subprocess.check_call(cmd)

    report_path = out_dir / "scene_pack_demo.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scene_pack"] == "demo"
    assert isinstance(report["inputs"], list)
    assert len(report["inputs"]) == 3
    for entry in report["inputs"]:
        assert "input_path" in entry
        assert "baseline" in entry and "candidate" in entry
        assert entry["baseline"]["run_dir"]
        assert entry["candidate"]["run_dir"]
        assert "metrics" in entry["baseline"]
        assert "metrics" in entry["candidate"]
        assert "diff" in entry


def test_scene_pack_determinism(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    _write_png(inputs_dir / "a.png", 5)
    _write_png(inputs_dir / "b.png", 15)
    crop_dir = inputs_dir / "crop1"
    crop_dir.mkdir()
    np.save(crop_dir / "crop.npy", np.zeros((4, 4), dtype=np.float32))
    (crop_dir / "meta.json").write_text(
        json.dumps(
            {
                "cfa_pattern": "RGGB",
                "black_level": 0.0,
                "white_level": 1.0,
                "bit_depth": 10,
                "x": 0,
                "y": 0,
                "w": 4,
                "h": 4,
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "-m",
        "mini_isp.tools.scene_pack",
        "--inputs",
        str(inputs_dir),
        "--out",
        str(out_dir),
        "--name",
        "demo",
        "--baseline-set",
        "stages.denoise.method=gaussian",
        "--candidate-set",
        "stages.denoise.method=chroma_gaussian",
    ]
    subprocess.check_call(cmd)
    report_a = json.loads((out_dir / "scene_pack_demo.json").read_text(encoding="utf-8"))

    # Run again to new output dir to avoid overwriting
    out_dir2 = tmp_path / "runs2"
    cmd2 = [
        sys.executable,
        "-m",
        "mini_isp.tools.scene_pack",
        "--inputs",
        str(inputs_dir),
        "--out",
        str(out_dir2),
        "--name",
        "demo",
        "--baseline-set",
        "stages.denoise.method=gaussian",
        "--candidate-set",
        "stages.denoise.method=chroma_gaussian",
    ]
    subprocess.check_call(cmd2)
    report_b = json.loads((out_dir2 / "scene_pack_demo.json").read_text(encoding="utf-8"))

    report_a["created_utc"] = "X"
    report_b["created_utc"] = "X"
    for entry in report_a["inputs"]:
        entry["baseline"]["run_dir"] = "X"
        entry["candidate"]["run_dir"] = "X"
        entry["baseline"]["runtime_ms"] = 0
        entry["candidate"]["runtime_ms"] = 0
    for entry in report_b["inputs"]:
        entry["baseline"]["run_dir"] = "X"
        entry["candidate"]["run_dir"] = "X"
        entry["baseline"]["runtime_ms"] = 0
        entry["candidate"]["runtime_ms"] = 0
    assert report_a == report_b


def test_scene_pack_skip_errors(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    _write_png(inputs_dir / "a.png", 5)
    bad_dir = inputs_dir / "bad_crop"
    bad_dir.mkdir()
    np.save(bad_dir / "crop.npy", np.zeros((2, 2), dtype=np.float32))

    out_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "-m",
        "mini_isp.tools.scene_pack",
        "--inputs",
        str(inputs_dir),
        "--out",
        str(out_dir),
        "--name",
        "demo",
        "--baseline-set",
        "stages.denoise.method=gaussian",
        "--candidate-set",
        "stages.denoise.method=chroma_gaussian",
        "--skip-errors",
    ]
    subprocess.check_call(cmd)
    report = json.loads((out_dir / "scene_pack_demo.json").read_text(encoding="utf-8"))
    assert len(report["inputs"]) == 1
