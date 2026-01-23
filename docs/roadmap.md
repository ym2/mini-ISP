# Roadmap

## v0.1 — single-frame reference + inspectability
- staged RAW→display output pipeline
- stable artifacts + `manifest.json` + static viewer
- modes: `classic`, `jdd`, `drc_plus`

## v0.2 — better baselines + comparisons
- improved denoise / diagnostics
- diagnostics examples: false-color/zipper checks, halo/ringing checks, simple stage-diff metrics
- side-by-side run comparison in viewer

## v0.3 — AI-DRC option (contained)
- `tone.method: ai_drc` (curve/LUT/coeffs)
- keep AI-DRC “predict curve/LUT + fast apply” (avoid full pixel-synthesis models at first)
- ONNX export hook (optional)

## v0.4+ — multi-frame wrappers
- MFNR/HDR orchestration as wrappers around the single-frame stages
