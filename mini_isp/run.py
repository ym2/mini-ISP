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
from .stages import StageResult, timed_call
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


def is_dng_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".dng"


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
            "input_ext": os.path.splitext(input_path)[1].lower(),
            "bit_depth": raw_meta.get("bit_depth"),
            "black_level": float(black_level),
            "white_level": float(white_level),
            "cfa_pattern": raw_meta.get("cfa_pattern", config["input"].get("bayer_pattern", "RGGB")),
            "wb_gains": raw_meta.get("wb_gains"),
            "wb_source": raw_meta.get("wb_source"),
            "daylight_wb_gains": raw_meta.get("daylight_wb_gains"),
            "cam_to_xyz_matrix": raw_meta.get("cam_to_xyz_matrix"),
            "cam_to_xyz_source": raw_meta.get("cam_to_xyz_source"),
            "xyz_to_working_matrix": raw_meta.get("xyz_to_working_matrix"),
            "xyz_to_working_source": raw_meta.get("xyz_to_working_source"),
            "non_dng_cam_to_xyz_matrix": raw_meta.get("non_dng_cam_to_xyz_matrix"),
            "non_dng_cam_to_xyz_source": raw_meta.get("non_dng_cam_to_xyz_source"),
            "non_dng_selected_input_variant": raw_meta.get("non_dng_selected_input_variant"),
            "non_dng_xyz_to_working_matrix_d65": raw_meta.get("non_dng_xyz_to_working_matrix_d65"),
            "non_dng_xyz_to_working_source_d65": raw_meta.get("non_dng_xyz_to_working_source_d65"),
            "non_dng_xyz_to_working_matrix_d50adapt": raw_meta.get("non_dng_xyz_to_working_matrix_d50adapt"),
            "non_dng_xyz_to_working_source_d50adapt": raw_meta.get("non_dng_xyz_to_working_source_d50adapt"),
            "non_dng_meta_reason": raw_meta.get("non_dng_meta_reason"),
            "ccm_auto_reason": raw_meta.get("ccm_auto_reason"),
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


def _resolve_wb_gains_stage_params(
    stage_config: Dict[str, Any],
    frame_meta: Dict[str, Any],
    cli_wb_mode: str | None,
    cli_wb_gains: List[float] | None,
) -> Dict[str, Any]:
    """
    Resolve wb_gains stage parameters, honoring precedence:
    - Explicit stage config wins (wb_gains/gains already set) -> treat as manual.
    - Else apply CLI override (meta/unity/manual) with defaults based on input kind.
    """
    out = dict(stage_config)

    if "wb_gains" in out or "gains" in out:
        # Explicit stage gains set by config; ignore CLI intent.
        out.setdefault("wb_mode", "manual")
        out.setdefault("wb_source", "manual")
        return out

    input_kind = str(frame_meta.get("input_kind", "png"))
    default_mode = "meta" if input_kind == "raw" else "unity"
    wb_mode = cli_wb_mode or default_mode

    if wb_mode == "manual":
        if not cli_wb_gains or len(cli_wb_gains) != 3:
            raise ValueError("wb_mode=manual requires wb_gains=[R,G,B]")
        out.update(
            {
                "wb_mode": "manual",
                "wb_source": "manual",
                "wb_gains": [float(cli_wb_gains[0]), float(cli_wb_gains[1]), float(cli_wb_gains[2])],
            }
        )
        return out

    if wb_mode == "unity":
        out.update({"wb_mode": "unity", "wb_source": "unity", "wb_gains": [1.0, 1.0, 1.0]})
        return out

    if wb_mode == "meta":
        wb_gains = frame_meta.get("wb_gains")
        wb_source = frame_meta.get("wb_source")
        if wb_gains is None:
            wb_gains = [1.0, 1.0, 1.0]
            wb_source = "unity_fallback"
        if wb_source == "unity":
            wb_source = "unity_fallback"
        out.update({"wb_mode": "meta", "wb_source": wb_source or "unity_fallback", "wb_gains": wb_gains})
        return out

    raise ValueError(f"Unknown wb_mode: {wb_mode}")


def _resolve_ccm_stage_params(
    stage_config: Dict[str, Any],
    frame_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve CCM stage parameters, honoring precedence:
    - Explicit CCM config (presence of behavior keys) always wins.
    - Otherwise, apply RAW auto-default policy from metadata when available:
      - DNG RAW: v0.2-M8 DNG-tags policy.
      - non-DNG RAW: v0.2-M9 non_dng_meta_default policy.
    - Non-RAW inputs keep default identity behavior.
    """
    out = dict(stage_config)
    out.setdefault("auto_default_applied", False)

    explicit_keys = {"mode", "matrix", "cam_to_xyz_matrix", "xyz_to_working_matrix"}
    if any(key in out for key in explicit_keys):
        out.setdefault("auto_default_reason", "explicit_stage_config")
        return out

    input_kind = str(frame_meta.get("input_kind", "png"))
    if input_kind != "raw":
        out.setdefault("auto_default_reason", "non_raw_input")
        return out

    input_ext = str(frame_meta.get("input_ext") or "").lower()
    source_path = str(frame_meta.get("source_path") or "")
    is_dng = input_ext == ".dng" or is_dng_path(source_path)
    if is_dng:
        cam_to_xyz = frame_meta.get("cam_to_xyz_matrix")
        xyz_to_working = frame_meta.get("xyz_to_working_matrix")
        cam_source = str(frame_meta.get("cam_to_xyz_source") or "none")
        xyz_source = str(frame_meta.get("xyz_to_working_source") or "none")
        if cam_to_xyz is None or xyz_to_working is None:
            out["cam_to_xyz_source"] = cam_source
            out["xyz_to_working_source"] = xyz_source
            out["auto_default_reason"] = str(frame_meta.get("ccm_auto_reason") or "missing_dng_ccm")
            return out

        out["mode"] = "chain"
        out["cam_to_xyz_matrix"] = cam_to_xyz
        out["xyz_to_working_matrix"] = xyz_to_working
        out["cam_to_xyz_source"] = cam_source if cam_source != "none" else "dng_tags_unavailable"
        out["xyz_to_working_source"] = xyz_source if xyz_source != "none" else "constant_xyz_d50_to_lin_srgb_d65"
        out["auto_default_applied"] = True
        out["auto_default_reason"] = "applied_from_dng_tags"
        return out

    # non-DNG RAW policy: non_dng_meta_default
    non_dng_cam_to_xyz = frame_meta.get("non_dng_cam_to_xyz_matrix")
    if non_dng_cam_to_xyz is None:
        out["cam_to_xyz_source"] = str(frame_meta.get("non_dng_cam_to_xyz_source") or "none")
        out["auto_default_reason"] = str(frame_meta.get("non_dng_meta_reason") or "missing_non_dng_meta")
        return out

    m_base = np.asarray(non_dng_cam_to_xyz, dtype=np.float32)
    if m_base.shape != (3, 3) or not np.all(np.isfinite(m_base)):
        out["auto_default_reason"] = "invalid_non_dng_cam_to_xyz"
        return out

    input_variant = str(frame_meta.get("non_dng_selected_input_variant") or "as_is")

    def _valid_wb_triplet(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            return None
        arr = arr[:3]
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            return None
        return arr

    daylight_wb = _valid_wb_triplet(frame_meta.get("daylight_wb_gains"))
    selected_wb = _valid_wb_triplet(frame_meta.get("wb_gains"))

    chosen_input_variant = "selected_input"
    chosen_wp_variant = "d50adapt"
    wb_for_unwb = selected_wb

    if daylight_wb is not None:
        chosen_input_variant = "pre_unwb_daylight"
        chosen_wp_variant = "d65"
        wb_for_unwb = daylight_wb

    m_cam_to_xyz = np.array(m_base, copy=True)
    if input_variant == "pre_unwb":
        if wb_for_unwb is None:
            out["auto_default_reason"] = "missing_non_dng_wb_for_pre_unwb"
            return out
        m_cam_to_xyz = (m_cam_to_xyz @ np.diag(1.0 / wb_for_unwb)).astype(np.float32)

    if chosen_wp_variant == "d65":
        xyz_to_working = frame_meta.get("non_dng_xyz_to_working_matrix_d65")
        xyz_source = str(frame_meta.get("non_dng_xyz_to_working_source_d65") or "constant_xyz_d65_to_lin_srgb_d65")
    else:
        xyz_to_working = frame_meta.get("non_dng_xyz_to_working_matrix_d50adapt")
        xyz_source = str(
            frame_meta.get("non_dng_xyz_to_working_source_d50adapt") or "constant_xyz_d50_to_lin_srgb_d65"
        )
    m_xyz_to_working = np.asarray(xyz_to_working, dtype=np.float32) if xyz_to_working is not None else None
    if m_xyz_to_working is None or m_xyz_to_working.shape != (3, 3) or not np.all(np.isfinite(m_xyz_to_working)):
        out["auto_default_reason"] = "invalid_non_dng_xyz_to_working"
        return out

    out["mode"] = "chain"
    out["cam_to_xyz_matrix"] = m_cam_to_xyz.tolist()
    out["xyz_to_working_matrix"] = m_xyz_to_working.tolist()
    out["cam_to_xyz_source"] = str(frame_meta.get("non_dng_cam_to_xyz_source") or "rawpy_rgb_xyz_matrix_non_dng")
    out["xyz_to_working_source"] = xyz_source
    out["ccm_source"] = "non_dng_meta_default"
    out["non_dng_meta_rule"] = "prefer_pre_unwb_daylight_d65_else_selected_d50adapt"
    out["non_dng_meta_input_variant"] = chosen_input_variant
    out["non_dng_meta_wp_variant"] = chosen_wp_variant
    out["auto_default_applied"] = True
    out["auto_default_reason"] = "applied_non_dng_meta_default"
    return out


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

    wb_cli = config.get("_cli", {}).get("wb", {})
    cli_wb_mode = wb_cli.get("mode")
    cli_wb_gains = wb_cli.get("gains")

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
        if stage.name == "wb_gains":
            stage_config = _resolve_wb_gains_stage_params(
                stage_config=stage_config,
                frame_meta=frame.meta,
                cli_wb_mode=cli_wb_mode,
                cli_wb_gains=cli_wb_gains,
            )
        if stage.name == "ccm":
            stage_config = _resolve_ccm_stage_params(stage_config=stage_config, frame_meta=frame.meta)

        if stage.name == "lsc" and stage_config.get("enabled") is False:
            result = StageResult(
                frame=Frame(image=np.array(frame.image, copy=True), meta=dict(frame.meta)),
                metrics={"skipped": True, "skip_reason": "disabled"},
            )
            timing_ms = 0.0
        else:
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
    parser.add_argument(
        "--wb-mode",
        choices=["meta", "unity", "manual"],
        default=None,
        help="Override wb_gains behavior: meta (RAW metadata), unity, or manual",
    )
    parser.add_argument(
        "--wb-gains",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "G", "B"),
        help="Manual WB gains as space-separated triple (R G B), used only with --wb-mode manual",
    )
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

    if args.wb_mode != "manual" and args.wb_gains is not None:
        parser.error("--wb-gains may only be used with --wb-mode manual")
    if args.wb_mode == "manual" and args.wb_gains is None:
        parser.error("--wb-gains R G B is required when --wb-mode manual")

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

    if args.wb_mode is not None:
        config.setdefault("_cli", {})
        config["_cli"].setdefault("wb", {})
        config["_cli"]["wb"]["mode"] = args.wb_mode
        if args.wb_mode == "manual":
            config["_cli"]["wb"]["gains"] = [float(x) for x in args.wb_gains]

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
