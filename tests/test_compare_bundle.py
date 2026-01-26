from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mini_isp.compare import match_stages, parse_compare_bundle


def test_parse_compare_bundle() -> None:
    bundle = {
        "schema_version": "0.1",
        "title": "Compare",
        "created_utc": "2026-01-01T00:00:00Z",
        "a": {"label": "A", "run_dir": "runs/a"},
        "b": {"label": "B", "run_dir": "runs/b"},
        "notes": "notes",
        "stage_match": {"primary": "index", "fallback": "name"},
    }
    parsed = parse_compare_bundle(bundle)
    assert parsed.schema_version == "0.1"
    assert parsed.a_label == "A"
    assert parsed.b_run_dir == "runs/b"
    assert parsed.stage_match_primary == "index"


def test_match_stages_index_then_name() -> None:
    stages_a = [
        {"index": 0, "name": "raw_norm"},
        {"index": 1, "name": "dpc"},
    ]
    stages_b = [
        {"index": 0, "name": "raw_norm"},
        {"index": 1, "name": "lsc"},
        {"index": 2, "name": "dpc"},
    ]
    matches = match_stages(stages_a, stages_b)
    assert matches[0]["a"]["name"] == "raw_norm"
    assert matches[0]["b"]["name"] == "raw_norm"
    assert matches[1]["a"]["name"] == "dpc"
    assert matches[1]["b"]["name"] == "dpc"
    assert matches[2]["a"] is None
    assert matches[2]["b"]["name"] == "lsc"


def test_compare_cli_generates_bundle(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "manifest.json").write_text("{}", encoding="utf-8")
    (run_b / "manifest.json").write_text("{}", encoding="utf-8")
    out_path = tmp_path / "compare.json"

    cmd = [
        sys.executable,
        "-m",
        "mini_isp.compare",
        "--a",
        str(run_a),
        "--b",
        str(run_b),
        "--out",
        str(out_path),
        "--label-a",
        "A",
        "--label-b",
        "B",
        "--notes",
        "notes",
        "--title",
        "title",
        "--created-utc",
        "2026-01-01T00:00:00Z",
    ]
    subprocess.check_call(cmd)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "0.1"
    assert data["title"] == "title"
    assert data["created_utc"] == "2026-01-01T00:00:00Z"
    assert data["a"]["run_dir"] == str(run_a)
    assert data["b"]["run_dir"] == str(run_b)
    assert data["stage_match"]["primary"] == "index"


def test_compare_cli_init(tmp_path: Path) -> None:
    out_path = tmp_path / "compare.json"
    cmd = [
        sys.executable,
        "-m",
        "mini_isp.compare",
        "--init",
        str(out_path),
    ]
    subprocess.check_call(cmd)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    for key in ["schema_version", "title", "created_utc", "a", "b", "notes", "stage_match"]:
        assert key in data
