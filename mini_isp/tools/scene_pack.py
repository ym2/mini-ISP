from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from mini_isp.run import DEFAULT_CONFIG, run_pipeline
from mini_isp.config_overrides import apply_overrides


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        try:
            import yaml

            return yaml.safe_load(f) or {}
        except Exception as exc:
            raise RuntimeError("PyYAML is required to load config YAML files") from exc


def _deep_copy_default() -> Dict[str, Any]:
    return json.loads(json.dumps(DEFAULT_CONFIG))


def _ensure_metrics(config: Dict[str, Any]) -> None:
    config.setdefault("metrics", {})
    config["metrics"]["enable"] = True
    config["metrics"]["diagnostics"] = True


def _extract_report_metrics(stage_dir: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics_path = os.path.join(stage_dir, "extra", "metrics.json")
    diff_path = os.path.join(stage_dir, "extra", "diff_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("min", "max", "p01", "p99", "luma_mean"):
            if key in data and isinstance(data[key], (int, float)):
                metrics[key] = float(data[key])
    if os.path.exists(diff_path):
        with open(diff_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("l1", "l2", "psnr"):
            if key in data and isinstance(data[key], (int, float)):
                metrics[key] = float(data[key])
    return metrics


def _sum_timing_ms(run_root: str) -> float:
    stages_root = os.path.join(run_root, "stages")
    total = 0.0
    for name in sorted(os.listdir(stages_root)):
        timing_path = os.path.join(stages_root, name, "timing_ms.json")
        if not os.path.exists(timing_path):
            continue
        with open(timing_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "timing_ms" in data:
            total += float(data["timing_ms"])
    return total


def _run_single(
    input_path: str,
    out_dir: str,
    name: str,
    config_path: Optional[str],
    set_overrides: List[str],
    enable_metrics: bool,
) -> Dict[str, Any]:
    config = _deep_copy_default()
    config["input"]["path"] = input_path
    config["output"]["dir"] = out_dir
    config["output"]["name"] = name

    if config_path:
        loaded = _load_config(config_path)
        from mini_isp.run import deep_update

        config = deep_update(config, loaded)
    if set_overrides:
        config = apply_overrides(config, set_overrides)
    if enable_metrics:
        _ensure_metrics(config)

    run_root = run_pipeline(config)
    stage_dirs = [
        os.path.join(run_root, "stages", d)
        for d in sorted(os.listdir(os.path.join(run_root, "stages")))
    ]
    metrics = {}
    if stage_dirs:
        metrics = _extract_report_metrics(stage_dirs[-1])
    runtime_ms = _sum_timing_ms(run_root)

    return {
        "run_dir": run_root,
        "pipeline_mode": config.get("pipeline_mode"),
        "runtime_ms": runtime_ms,
        "metrics": metrics,
    }


def _collect_inputs(inputs_dir: str, exts: List[str], skip_errors: bool) -> List[str]:
    entries: List[str] = []
    for root, _, files in os.walk(inputs_dir):
        for name in files:
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            if exts and ext not in exts:
                continue
            if ext == ".npy":
                meta_path = os.path.join(os.path.dirname(path), "meta.json")
                if not os.path.exists(meta_path):
                    if skip_errors:
                        continue
                    raise FileNotFoundError(f"meta.json not found next to {path}")
            entries.append(path)
    return sorted(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scene-pack runner with consolidated JSON report.")
    parser.add_argument("--inputs", required=True, help="Input directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--name", default="scene_pack", help="Report name stem")
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--candidate-config", default=None)
    parser.add_argument("--baseline-set", action="append", default=[])
    parser.add_argument("--candidate-set", action="append", default=[])
    metrics_group = parser.add_mutually_exclusive_group()
    metrics_group.add_argument("--metrics", action="store_true", default=True)
    metrics_group.add_argument("--no-metrics", action="store_false", dest="metrics")
    parser.add_argument("--ext", default=".png,.dng,.nef,.cr2,.arw,.npy")
    parser.add_argument("--skip-errors", action="store_true", help="Skip inputs that fail validation")
    args = parser.parse_args()

    if not args.baseline_config and not args.baseline_set:
        raise ValueError("Baseline config is required (config path or --baseline-set)")
    if not args.candidate_config and not args.candidate_set:
        raise ValueError("Candidate config is required (config path or --candidate-set)")

    exts = [e.strip().lower() for e in args.ext.split(",") if e.strip()]
    inputs = _collect_inputs(args.inputs, exts, args.skip_errors)
    os.makedirs(args.out, exist_ok=True)

    report_inputs = []
    for path in inputs:
        stem = os.path.splitext(os.path.basename(path))[0]
        base_name = os.path.join(args.name, f"{stem}__baseline")
        cand_name = os.path.join(args.name, f"{stem}__candidate")
        baseline = _run_single(
            path,
            args.out,
            base_name,
            args.baseline_config,
            args.baseline_set,
            args.metrics,
        )
        candidate = _run_single(
            path,
            args.out,
            cand_name,
            args.candidate_config,
            args.candidate_set,
            args.metrics,
        )

        diff = {}
        for key in baseline["metrics"]:
            if key in candidate["metrics"]:
                diff[f"{key}_delta"] = candidate["metrics"][key] - baseline["metrics"][key]

        report_inputs.append(
            {
                "input_path": path,
                "baseline": {
                    "label": "baseline",
                    "run_dir": baseline["run_dir"],
                    "pipeline_mode": baseline["pipeline_mode"],
                "runtime_ms": baseline["runtime_ms"],
                    "metrics": baseline["metrics"],
                },
                "candidate": {
                    "label": "candidate",
                    "run_dir": candidate["run_dir"],
                    "pipeline_mode": candidate["pipeline_mode"],
                "runtime_ms": candidate["runtime_ms"],
                    "metrics": candidate["metrics"],
                },
                "diff": diff,
            }
        )

    report = {
        "scene_pack": args.name,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "inputs": report_inputs,
    }
    out_path = os.path.join(args.out, f"scene_pack_{args.name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
