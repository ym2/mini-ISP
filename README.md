# mini-ISP

**A staged RAW→display output reference ISP pipeline with per-stage dumps, metrics, and an interactive viewer for image-quality (IQ) debugging and learning.**

mini-ISP is a compact, modular ISP pipeline you can run on RAW images, inspect stage-by-stage outputs, and compare algorithm choices with reproducible experiments. It’s a structured, hands-on way to explore and strengthen ISP and camera IQ skills (some familiarity with imaging pipelines helps).

## What you get in v0.1
- **End-to-end RAW→display output pipeline** with clear, swappable stages and strict I/O contracts
- **Core ISP stages implemented**: RAW normalization → DPC → LSC → WB → demosaic → denoise → CCM → 3A stats → tone → color adjust → sharpen → OETF encode
- **Pipeline modes**: `classic` (separate stages), `jdd` (joint demosaic+denoise), `drc_plus` (tone/DRC + color adjustment/preservation)
- **Inspectability built-in**: per-stage previews + ROI crops + debug JSON + timings/metrics for every run
- **Static HTML stage viewer** powered by a stable `manifest.json` and predictable asset layout
- **Optional skin-mask hook** (off by default): heuristic now, upgrade path to seg-model later

## Pipeline modes
- `classic`: separate stages (best for learning/debugging)
- `jdd`: joint demosaic+denoise stage (curated composite)
- `drc_plus`: tone/DRC + color adjustment/preservation (curated composite)

See: [`docs/pipeline.md`](docs/pipeline.md)

## Quickstart
> This will be updated as the codebase lands. For now, the goal is: one command → a run folder with stage artifacts + viewer.

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# pip install -r requirements.txt   # (once added)
# python -m mini_isp.run --config configs/default.yaml --pipeline_mode classic --input data/sample.dng --out runs