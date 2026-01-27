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

## v0.2 — comparisons + diagnostics + better baselines
- make improvements *measurable* and easy to review (A/B comparison + metrics)
- add optional diagnostics outputs (no breaking changes to v0.1 run layout)
- upgrade a few baselines once we can compare reliably

### Milestones (v0.2)
All v0.2 work must be **backward-compatible** with v0.1 runs:
- do **not** break `manifest.json` schema or viewer asset paths
- any new outputs go under stage folders (e.g., `stages/<nn>_<name>/extra/`) or as additional optional JSON files

- **M1 — Compare bundle + side-by-side viewer**
  - Add a small *compare bundle* (e.g., `compare.json`) that points to two manifests (A/B), labels, and optional notes.
  - Extend the static viewer to load **one manifest (default)** or **two manifests (compare mode)** and show side-by-side previews with synced stage selection.
  - Done when: viewer works for single-run as before, and also compares two runs without breaking old runs.

- **M2 — Stage-diff metrics + diagnostics outputs**
  - Add simple per-stage / stage-diff metrics (e.g., L1/L2 diff on previews; optional metrics on linear buffers if available).
  - Add optional diagnostics artifacts (examples): false-color/zipper check, halo/ringing check, simple “before/after” diffs.
  - Done when: metrics/diagnostics are emitted only when enabled and tests assert files/keys exist (no schema break).

- **M3 — Denoise upgrade (demonstrable improvement)**
  - Improve `denoise` quality without heavy deps (beyond the v0.1 Gaussian/box baseline); recommended: chroma-aware Gaussian or simple edge-aware strength.
  - Done when: on deterministic synthetic noise tests, the new method improves objective error (MSE/PSNR) vs the current Gaussian baseline and still preserves dtype/shape/contracts.

- **M4 — Sharpen tuning baseline (prove improvement)**
  - Goal: improve sharpen and verify improvement with A/B compare + diagnostics/metrics.
  - Deliverables:
    - add a tuned `sharpen` variant (keep existing `unsharp_mask` intact; add minimal new params only if needed)
    - use compare bundle + metrics + diagnostics to show improvement (e.g., edge contrast up, halo proxy down)
    - keep defaults conservative and explain via debug fields
  - Acceptance:
    - improvement shown in an A/B compare bundle
    - no regressions in tests (`pytest -q` passes)
    - still minimal deps; PNG bootstrap still works

- **M5 — Optional real RAW/DNG input support (for validation)**
  - Add optional RAW/DNG loading so the pipeline can run on a true Bayer mosaic (keep PNG bootstrap unchanged).
  - Keep deps minimal; allow RAW support to be an optional install (not required for default PNG runs).
  - Ensure CFA pattern and normalization metadata propagate (e.g., `meta.cfa_pattern`, manifest `input.cfa_pattern`).
  - Done when: PNG runs behave exactly as before; a real RAW/DNG run completes end-to-end with the same run-folder layout/artifacts and tests cover basic RAW normalization + CFA propagation.
  
Notes:
- v0.2 can be developed using PNG-bootstrap runs; M5 adds RAW/DNG input support to make diagnostics and A/B comparisons more representative on real sensor data.

## v0.3 — AI-DRC option (contained)
- `tone.method: ai_drc` (curve/LUT/coeffs)
- keep AI-DRC “predict curve/LUT + fast apply” (avoid full pixel-synthesis models at first)
- ONNX export hook (optional)

## v0.4+ — multi-frame wrappers
- MFNR/HDR orchestration as wrappers around the single-frame stages
