from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from .artifacts import write_stage_artifacts
from .io_utils import (
    Frame,
    ensure_dir,
    load_png_as_rgb,
    save_png,
    to_display_u8,
    write_json,
    write_yaml,
)
from .pipeline import build_pipeline
from .stages import timed_call


DEFAULT_CONFIG: Dict[str, Any] = {
    "pipeline_mode": "classic",
    "input": {
        "path": "data/sample.png",
        "bayer_pattern": "RGGB",
        "bit_depth": 8,
        "black_level": 0,
        "white_level": 255,
    },
    "output": {"dir": "runs", "name": None, "save_final": True, "format": "png", "bit_depth": 8},
    "dump": {
        "enable": True,
        "preview_max_side": 1024,
        "roi": {"enable": True, "xywh": [0.35, 0.35, 0.30, 0.30]},
        "stages": "all",
    },
    "metrics": {"enable": True, "timing": True, "histograms": False, "deltas": False},
    "skin_mask": {"enable": False, "method": "heuristic", "dump": True},
    "stages": {},
    "viewer": {"enable": True, "title": "mini-ISP run"},
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base.get(key, {}), value)
        else:
            base[key] = value
    return base


def load_config(path: str | None) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        loaded = None
        has_yaml = False
        try:
            import yaml
            has_yaml = True
        except Exception:
            has_yaml = False
        if not has_yaml:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
            except Exception as exc:
                raise RuntimeError("PyYAML is required to load config YAML files") from exc
        else:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
        config = deep_update(config, loaded or {})
    return config


def maybe_generate_sample(path: str) -> None:
    if os.path.exists(path):
        return
    ensure_dir(os.path.dirname(path))
    width, height = 640, 480
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xv = np.tile(x[None, :], (height, 1))
    yv = np.tile(y, (1, width))
    gradient = np.stack([xv, yv, 0.5 * np.ones((height, width), dtype=np.float32)], axis=2)
    # Color bars on bottom 1/4
    bars = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    bar_h = height // 4
    bar_w = width // bars.shape[0]
    for i, color in enumerate(bars):
        x0 = i * bar_w
        x1 = width if i == bars.shape[0] - 1 else (i + 1) * bar_w
        gradient[height - bar_h : height, x0:x1, :] = color
    save_png(path, (gradient * 255.0 + 0.5).astype(np.uint8))


def resolve_run_id(name: str | None) -> str:
    if name:
        return name
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def should_dump_stage(config: Dict[str, Any], stage_name: str) -> bool:
    if not config["dump"]["enable"]:
        return False
    stages = config["dump"].get("stages", "all")
    return stages == "all" or stage_name in stages


def stage_dir_name(index: int, stage_name: str) -> str:
    return f"{index:02d}_{stage_name}"


def run_pipeline(config: Dict[str, Any]) -> str:
    input_path = config["input"]["path"]
    maybe_generate_sample(input_path)
    image_rgb = load_png_as_rgb(input_path)
    meta = {
        "source_path": input_path,
        "bit_depth": config["input"].get("bit_depth", 8),
        "black_level": config["input"].get("black_level", 0),
        "white_level": config["input"].get("white_level", 255),
        "cfa_pattern": config["input"].get("bayer_pattern", "RGGB"),
    }
    frame = Frame(image=image_rgb, meta=meta)

    run_id = resolve_run_id(config["output"].get("name"))
    run_root = os.path.join(config["output"]["dir"], run_id)
    ensure_dir(run_root)
    ensure_dir(os.path.join(run_root, "stages"))
    ensure_dir(os.path.join(run_root, "final"))
    ensure_dir(os.path.join(run_root, "viewer"))

    write_yaml(os.path.join(run_root, "config_resolved.yaml"), config)

    manifest: Dict[str, Any] = {
        "schema_version": "0.1",
        "run_id": run_id,
        "title": config.get("viewer", {}).get("title", "mini-ISP run"),
        "input": {
            "path": input_path,
            "width": int(image_rgb.shape[1]),
            "height": int(image_rgb.shape[0]),
            "cfa_pattern": meta.get("cfa_pattern"),
        },
        "pipeline_mode": config["pipeline_mode"],
        "final": {"path": "final/output.png"},
        "stages": [],
    }

    pipeline = build_pipeline(config["pipeline_mode"])

    for index, stage in enumerate(pipeline):
        stage_dir = stage_dir_name(index, stage.name)
        stage_root = os.path.join(run_root, "stages", stage_dir)
        ensure_dir(stage_root)
        stage_config = config.get("stages", {}).get(stage.name, {})
        # Pass through stage-specific params; raw_norm expects input overrides
        if stage.name == "raw_norm":
            stage_config = {
                **stage_config,
                "cfa_pattern": meta.get("cfa_pattern"),
                "black_level": meta.get("black_level"),
                "white_level": meta.get("white_level"),
            }

        result, timing_ms = timed_call(stage.run, frame, stage_config)
        frame = result.frame

        if should_dump_stage(config, stage.name):
            write_stage_artifacts(
                stage_root=stage_root,
                stage_name=stage.name,
                image=frame.image,
                stage_params=stage_config,
                metrics=result.metrics,
                timing_ms=timing_ms,
                dump_config=config["dump"],
            )

        artifacts = {
            "preview": f"stages/{stage_dir}/preview.png",
            "debug": f"stages/{stage_dir}/debug.json",
            "timing": f"stages/{stage_dir}/timing_ms.json",
        }
        if config["dump"]["roi"]["enable"]:
            artifacts["roi"] = f"stages/{stage_dir}/roi.png"
        manifest["stages"].append(
            {
                "index": index,
                "name": stage.name,
                "display_name": stage.display_name,
                "dir": f"stages/{stage_dir}",
                "artifacts": artifacts,
                "timing_ms": timing_ms,
            }
        )

    # Final output
    final_path = os.path.join(run_root, "final", "output.png")
    final_u8 = to_display_u8(frame.image)
    save_png(final_path, final_u8)

    write_json(os.path.join(run_root, "manifest.json"), manifest)
    copy_viewer_assets(os.path.join(run_root, "viewer"))

    return run_root


def copy_viewer_assets(dst_dir: str) -> None:
    import importlib.resources as resources

    package = "mini_isp.viewer_assets"
    for name in ["index.html", "app.js", "styles.css"]:
        with resources.open_text(package, name, encoding="utf-8") as f:
            contents = f.read()
        with open(os.path.join(dst_dir, name), "w", encoding="utf-8") as out:
            out.write(contents)


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-ISP runner (v0.1)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--input", type=str, default=None, help="Override input path")
    parser.add_argument("--out", type=str, default=None, help="Override output directory")
    parser.add_argument("--pipeline_mode", type=str, default=None, help="Override pipeline mode")
    parser.add_argument("--name", type=str, default=None, help="Override run id")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.input:
        config["input"]["path"] = args.input
    if args.out:
        config["output"]["dir"] = args.out
    if args.pipeline_mode:
        config["pipeline_mode"] = args.pipeline_mode
    if args.name:
        config["output"]["name"] = args.name

    run_root = run_pipeline(config)
    print(run_root)


if __name__ == "__main__":
    main()
