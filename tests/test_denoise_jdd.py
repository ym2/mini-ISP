from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mini_isp.io_utils import Frame
from mini_isp.pipeline import build_pipeline
from mini_isp.run import DEFAULT_CONFIG, run_pipeline, stage_dir_name
from mini_isp.stages import stage_denoise


def test_denoise_reduces_noise() -> None:
    h, w = 32, 32
    clean = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    clean_rgb = np.stack([clean, clean, clean], axis=2).astype(np.float32)
    rng = np.random.default_rng(0)
    noisy = clean_rgb + rng.normal(0.0, 0.05, size=clean_rgb.shape).astype(np.float32)

    frame = Frame(image=noisy, meta={})
    result = stage_denoise(frame, {"method": "gaussian", "sigma": 1.0, "ksize": 5})
    denoised = result.frame.image

    mse_before = float(np.mean((noisy - clean_rgb) ** 2))
    mse_after = float(np.mean((denoised - clean_rgb) ** 2))

    assert denoised.shape == noisy.shape
    assert denoised.dtype == np.float32
    assert np.isfinite(denoised).all()
    assert mse_after < mse_before


def test_chroma_gaussian_improves_over_gaussian() -> None:
    h, w = 32, 32
    clean = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    clean_rgb = np.stack([clean, clean, clean], axis=2).astype(np.float32)
    rng = np.random.default_rng(123)
    noisy = clean_rgb + rng.normal(0.0, 0.05, size=clean_rgb.shape).astype(np.float32)
    frame = Frame(image=noisy, meta={})

    gaussian = stage_denoise(frame, {"method": "gaussian", "sigma": 1.0, "ksize": 5}).frame.image
    chroma = stage_denoise(
        frame,
        {"method": "chroma_gaussian", "sigma_y": 1.0, "sigma_c": 2.0, "ksize": 5},
    ).frame.image

    mse_gauss = float(np.mean((gaussian - clean_rgb) ** 2))
    mse_chroma = float(np.mean((chroma - clean_rgb) ** 2))
    assert chroma.shape == gaussian.shape
    assert chroma.dtype == np.float32
    assert np.isfinite(chroma).all()
    assert mse_chroma < mse_gauss


def test_jdd_pipeline_artifacts(tmp_path: Path) -> None:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["input"]["path"] = str(tmp_path / "sample.png")
    config["output"]["dir"] = str(tmp_path / "runs")
    config["output"]["name"] = "jdd_smoke"
    config["pipeline_mode"] = "jdd"
    config["dump"]["enable"] = True
    config["dump"]["roi"]["enable"] = True

    run_root = Path(run_pipeline(config))
    stages_root = run_root / "stages"
    pipeline = build_pipeline("jdd")
    for index, stage in enumerate(pipeline):
        stage_dir = stages_root / stage_dir_name(index, stage.name)
        assert (stage_dir / "preview.png").exists()
        assert (stage_dir / "debug.json").exists()
        assert (stage_dir / "timing_ms.json").exists()
        assert (stage_dir / "roi.png").exists()
