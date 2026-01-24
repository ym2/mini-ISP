# Roadmap

## v0.1 — single-frame reference + inspectability
- staged RAW→display output pipeline
- stable artifacts + `manifest.json` + static viewer
- modes: `classic`, `jdd`, `drc_plus`

### Milestones (v0.1)
Each milestone should end with: **one runnable command** → a valid `runs/<run_id>/` folder, a correct `manifest.json`, and the viewer loading stage previews.

- **M1 — Bootstrap runner + manifest + viewer**
  - Done when: `python -m mini_isp.run ...` works on a PNG input (generated if missing) and produces the full run folder + viewer.
- **M2 — Stage interface + artifacts helpers + smoke tests**
  - Done when: stubs for all v0.1 stages emit required artifacts and `pytest -q` passes a basic end-to-end smoke test.
- **M3 — RAW-domain realism: DPC + LSC**
  - Done when: DPC fixes synthetic defects; LSC applies a gain map (radial baseline) with a configurable cap; tests cover counts/caps.
- **M4 — WB + demosaic (RAW→linear RGB)**
  - Done when: WB gains are applied consistently and demosaic produces `RGB_LINEAR_F32` with range checks + artifacts.
- **M5 — Denoise baseline + `jdd` composite**
  - Done when: `classic` and `jdd` both run end-to-end; `jdd` is a curated composite that preserves compatible I/O + artifacts.
- **M6 — CCM + 3A stats**
  - Done when: CCM (identity/manual/profile) and 3A stats dumps are present and stable across runs; tests assert required meta keys.
- **M7 — Tone + color adjust + sharpen + OETF encode**
  - Done when: the pipeline produces a display-ready final output and per-stage previews look sensible on real images.
- **M8 — `drc_plus` curated composite**
  - Done when: `drc_plus` runs end-to-end and reuses core tone/curve-apply code paths where possible.

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
