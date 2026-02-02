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
    load_raw_mosaic,
    normalize_raw_mosaic,
    save_png,
    to_display_u8,
    write_json,
    write_yaml,
)
from .pipeline import build_pipeline
from .stages import timed_call
from .metrics import build_preview_for_metrics, emit_metrics_and_diagnostics
from .config_overrides import apply_overrides


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
    "metrics": {"enable": False, "diagnostics": False, "timing": True, "histograms": False, "deltas": False},
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


def should_emit_metrics(config: Dict[str, Any]) -> bool:
    return bool(config.get("metrics", {}).get("enable", False))


def should_emit_diagnostics(config: Dict[str, Any]) -> bool:
    return bool(config.get("metrics", {}).get("diagnostics", False))


def stage_dir_name(index: int, stage_name: str) -> str:
    return f"{index:02d}_{stage_name}"


def is_raw_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".dng", ".nef", ".cr2", ".arw", ".rw2", ".orf", ".raf", ".raw"}


def load_npy_mosaic(path: str) -> Frame:
    meta_path = os.path.join(os.path.dirname(path), "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("meta.json not found next to crop.npy")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    required = ["cfa_pattern", "black_level", "white_level", "bit_depth", "x", "y", "w", "h"]
    missing = [key for key in required if key not in meta_json]
    if missing:
        raise ValueError(f"meta.json missing required keys: {', '.join(missing)}")
    mosaic = np.load(path)
    if mosaic.ndim != 2:
        raise ValueError("crop.npy must be a 2D mosaic")
    if mosaic.dtype == np.float32:
        mosaic_norm = np.clip(mosaic, 0.0, 1.0).astype(np.float32)
    elif mosaic.dtype == np.uint16:
        mosaic_norm = normalize_raw_mosaic(
            mosaic, float(meta_json["black_level"]), float(meta_json["white_level"])
        )
    else:
        raise ValueError("crop.npy dtype must be float32 or uint16")

    meta = {
        "source_path": path,
        "bit_depth": meta_json.get("bit_depth"),
        "black_level": float(meta_json["black_level"]),
        "white_level": float(meta_json["white_level"]),
        "cfa_pattern": meta_json.get("cfa_pattern", "RGGB"),
        "raw_mosaic": True,
        "input_kind": "npy",
    }
    return Frame(image=mosaic_norm, meta=meta)


def load_input_frame(config: Dict[str, Any]) -> Frame:
    input_path = config["input"]["path"]
    if input_path.lower().endswith(".npy"):
        return load_npy_mosaic(input_path)
    if is_raw_path(input_path):
        mosaic, raw_meta = load_raw_mosaic(input_path, config["input"].get("bayer_pattern", "RGGB"))
        black_level = raw_meta.get("black_level", 0.0)
        white_level = raw_meta.get("white_level")
        if white_level is None:
            if np.issubdtype(mosaic.dtype, np.integer):
                white_level = float(np.iinfo(mosaic.dtype).max)
            else:
                white_level = float(np.max(mosaic))
        mosaic_norm = normalize_raw_mosaic(mosaic, float(black_level), float(white_level))
        meta = {
            "source_path": input_path,
            "bit_depth": raw_meta.get("bit_depth"),
            "black_level": float(black_level),
            "white_level": float(white_level),
            "cfa_pattern": raw_meta.get("cfa_pattern", config["input"].get("bayer_pattern", "RGGB")),
            "wb_gains": raw_meta.get("wb_gains"),
            "wb_source": raw_meta.get("wb_source"),
            "raw_mosaic": True,
            "input_kind": "raw",
        }
        return Frame(image=mosaic_norm, meta=meta)

    maybe_generate_sample(input_path)
    image_rgb = load_png_as_rgb(input_path)
    meta = {
        "source_path": input_path,
        "bit_depth": config["input"].get("bit_depth", 8),
        "black_level": config["input"].get("black_level", 0),
        "white_level": config["input"].get("white_level", 255),
        "cfa_pattern": config["input"].get("bayer_pattern", "RGGB"),
        "raw_mosaic": False,
        "input_kind": "png",
    }
    return Frame(image=image_rgb, meta=meta)


def run_pipeline(config: Dict[str, Any]) -> str:
    input_path = config["input"]["path"]
    frame = load_input_frame(config)

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
            "width": int(frame.image.shape[1]),
            "height": int(frame.image.shape[0]),
            "cfa_pattern": frame.meta.get("cfa_pattern"),
        },
        "pipeline_mode": config["pipeline_mode"],
        "final": {"path": "final/output.png"},
        "stages": [],
    }

    if frame.meta.get("input_kind") == "raw":
        for key in ("bit_depth", "black_level", "white_level"):
            value = frame.meta.get(key)
            if value is not None:
                manifest["input"][key] = value
        wb_gains = frame.meta.get("wb_gains")
        if wb_gains is not None:
            manifest["input"]["wb_gains"] = wb_gains

    pipeline = build_pipeline(config["pipeline_mode"])

    prev_preview = None
    for index, stage in enumerate(pipeline):
        stage_dir = stage_dir_name(index, stage.name)
        stage_root = os.path.join(run_root, "stages", stage_dir)
        ensure_dir(stage_root)
        stage_config = config.get("stages", {}).get(stage.name, {})
        # Pass through stage-specific params; raw_norm expects input overrides
        if stage.name == "raw_norm":
            stage_config = {
                **stage_config,
                "cfa_pattern": frame.meta.get("cfa_pattern"),
                "black_level": frame.meta.get("black_level"),
                "white_level": frame.meta.get("white_level"),
            }
        if stage.name == "wb_gains" and "wb_gains" not in stage_config:
            wb_gains = frame.meta.get("wb_gains")
            if wb_gains is not None:
                stage_config = {
                    **stage_config,
                    "wb_gains": wb_gains,
                    "wb_source": frame.meta.get("wb_source", "unity"),
                }

        result, timing_ms = timed_call(stage.run, frame, stage_config)
        frame = result.frame

        preview = None
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
            preview = build_preview_for_metrics(stage_root, config)

        if should_emit_metrics(config) and preview is not None:
            emit_metrics_and_diagnostics(
                stage_root=stage_root,
                preview=preview,
                prev_preview=prev_preview,
                enable_diagnostics=should_emit_diagnostics(config),
                metrics_out=config.get("metrics", {}).get("out", "extra"),
            )
        if preview is not None:
            prev_preview = preview

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
    parser.add_argument("--enable-metrics", action="store_true", help="Enable metrics outputs")
    parser.add_argument("--enable-diagnostics", action="store_true", help="Enable diagnostics outputs")
    parser.add_argument(
        "--metrics-target",
        choices=["preview", "linear", "both"],
        default="preview",
        help="Metrics target domain",
    )
    parser.add_argument(
        "--metrics-diff",
        choices=["off", "l1", "l2", "psnr", "all"],
        default="l1",
        help="Diff metric selection",
    )
    parser.add_argument(
        "--metrics-out",
        choices=["stage_root", "extra"],
        default="extra",
        help="Metrics output location",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config via KEY=VALUE (repeatable)",
    )
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

    if args.overrides:
        config = apply_overrides(config, args.overrides)

    if args.metrics_target in ("linear", "both"):
        raise SystemExit("linear metrics not implemented yet; use --metrics-target preview")

    if args.enable_metrics:
        config.setdefault("metrics", {})
        config["metrics"]["enable"] = True
        config["metrics"]["target"] = args.metrics_target
        config["metrics"]["diff"] = args.metrics_diff
        config["metrics"]["out"] = args.metrics_out
    if args.enable_diagnostics:
        config.setdefault("metrics", {})
        config["metrics"]["diagnostics"] = True

    run_root = run_pipeline(config)
    print(run_root)


if __name__ == "__main__":
    main()
