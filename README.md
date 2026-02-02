# mini-ISP

**A staged RAW→display reference ISP pipeline built for inspectability, reproducible experiments, and A/B comparisons.**

mini-ISP is a compact, modular ISP pipeline you can run on RAW or PNG bootstrap inputs, inspect stage-by-stage artifacts, and compare algorithm choices with a manifest-driven static viewer. It’s designed as an evolving sandbox for learning and validating camera IQ/ISP ideas—from single-frame baselines to future multi-frame and learned modules—while keeping outputs deterministic and easy to review.

## Key features (current)
- **Deterministic runs**: stable `runs/<run_id>/` layout with a canonical `manifest.json`.
- **Per-stage inspectability**: each stage emits `preview.png`, `debug.json`, `timing_ms.json`, and optional `roi.png`.
- **Static viewer**: step through stages; view preview/ROI/debug/timing; supports **single-run** and **A/B compare** modes.
- **Diagnostics/metrics surfaced in viewer**: metrics tables and diagnostics toggles when outputs exist.
- **Optional diagnostics/metrics outputs**: additive files under `stages/<nn>_<name>/extra/` (no schema breaks).
- **Config + CLI overrides**: YAML config plus `--set KEY=VALUE` overrides and flags to enable metrics/diagnostics.
- **Minimal dependencies**: NumPy + Pillow/PyYAML; optional RAW support via `rawpy`.

## Pipeline overview
Single-frame staged pipeline (v0.1 complete), with curated composites:
- `classic`: separate stages (best for learning/debugging)
- `jdd`: composite raw→rgb wrapper (demosaic + denoise)
- `drc_plus`: composite tone + color wrapper

See: [`docs/pipeline.md`](docs/pipeline.md) and [`docs/stage_contracts.md`](docs/stage_contracts.md).

## Install
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional RAW support (requires rawpy):
```bash
pip install -r requirements-raw.txt
```

Dev/test deps:
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Single-run workflow
Serve the viewer over HTTP (some browsers block `fetch()` on `file://`):
```bash
python -m http.server 8000
```

### PNG bootstrap (quick start / no RAW dependency)
If `data/sample.png` is missing, the runner will generate a deterministic sample image.

```bash
python -m mini_isp.run   --input data/sample.png   --out runs   --pipeline_mode classic   --name png_demo
```

Open:
```
http://localhost:8000/runs/png_demo/viewer/index.html?manifest=/runs/png_demo/manifest.json
```

### Run on a real RAW (recommended; requires rawpy)
```bash
python -m mini_isp.run   --input path/to/sample.dng   --out runs   --pipeline_mode classic   --name raw_demo
```
Supported RAW extensions depend on rawpy/LibRaw (e.g., dng/nef/cr2/arw/rw2/orf/raf).

Open:
```
http://localhost:8000/runs/raw_demo/viewer/index.html?manifest=/runs/raw_demo/manifest.json
```

### Run from a RAW crop (crop.npy + meta.json)
```bash
python -m mini_isp.run   --input crops/raw_demo/crop.npy   --out runs   --pipeline_mode classic   --name crop_demo
```

Open:
```
http://localhost:8000/runs/crop_demo/viewer/index.html?manifest=/runs/crop_demo/manifest.json
```

## A/B compare workflow
1) Create two runs (example: different denoise methods):
```bash
python -m mini_isp.run --input data/sample.png --out runs --pipeline_mode classic --name A   --set stages.denoise.method=gaussian

python -m mini_isp.run --input data/sample.png --out runs --pipeline_mode classic --name B   --set stages.denoise.method=chroma_gaussian
```

2) Generate a compare bundle JSON:
```bash
python -m mini_isp.compare   --a runs/A   --b runs/B   --out runs/compare/compare.json   --label-a "Baseline"   --label-b "New"   --notes "denoise A/B"
```

3) Start a server from the **repo root** and open compare mode:
```bash
python -m http.server 8000
# open: http://localhost:8000/runs/A/viewer/index.html?compare=/runs/compare/compare.json
```

## Metrics / diagnostics
Enable metrics and diagnostics outputs (files are written under `stages/.../extra/` and `stages/.../extra/diagnostics/`):
```bash
python -m mini_isp.run --input data/sample.png --out runs --pipeline_mode classic --name metrics_demo   --enable-metrics --enable-diagnostics
```

Viewer usage: the Metrics panel and Diagnostics toggles appear when these files exist. The Metrics panel defaults to a small subset and includes a “Show all” toggle for the full list.

## Tone (parametric Reinhard)
Tune Reinhard with `--set` overrides:
```bash
python -m mini_isp.run --input data/sample.png --out runs --pipeline_mode classic --name tone_demo \
  --set stages.tone.method=reinhard \
  --set stages.tone.exposure=1.0 \
  --set stages.tone.white_point=1.0 \
  --set stages.tone.gamma=1.0
```
Replace the parameter values above with your desired settings.

## RAW crop utility (testing support)
Generate small, deterministic Bayer crops for fast iteration and reproducible A/B tests:
```bash
python -m mini_isp.tools.raw_crop --input path/to/sample.dng --out crops/raw_demo --x 100 --y 200 --w 512 --h 512 --dtype float32
```
Use the resulting `crop.npy` + `meta.json` as a pipeline input (see “Run from a RAW crop” above).

## Scene-pack runner (batch A/B + report)
Run a folder of scenes through baseline vs candidate configs and emit a consolidated JSON report:
```bash
python -m mini_isp.tools.scene_pack \
  --inputs data/scenes \
  --out runs/scene_pack_demo \
  --name denoise_tweak_01 \
  --baseline-set stages.denoise.method=gaussian \
  --candidate-set stages.denoise.method=chroma_gaussian
```
Report output:
```
runs/scene_pack_demo/scene_pack_denoise_tweak_01.json
```
If your inputs include `crop.npy`, ensure each crop lives in its own folder with a sibling `meta.json`. Use `--skip-errors` to ignore invalid inputs.

## Repo conventions
- **No breaking changes** to `manifest.json` schema or viewer asset paths (compatibility matters).
- New optional outputs go under `stages/<nn>_<name>/extra/` or additional optional JSON files.

## Roadmap
- **v0.3**: AI-DRC option (`tone.method: ai_drc`) with “predict curve/LUT + fast apply”; optional ONNX hook.
- **v0.4+**: multi-frame wrappers (MFNR/HDR orchestration around the single-frame stages).

See: [`roadmap.md`](docs/roadmap.md).

## License
See [`LICENSE`](LICENSE).
