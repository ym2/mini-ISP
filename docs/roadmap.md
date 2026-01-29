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

- **M6 — RAW crop utility (testing support)**
  - Add a small, optional utility to generate deterministic Bayer RAW crops for faster ISP iteration and validation.
  - Support cropping from real RAW inputs using the same RAW loader as the pipeline (e.g., rawpy), without modifying the ISP run flow.
  - Output a cropped Bayer mosaic plus minimal metadata; add a dev-only input path so mini_isp.run can load crop.npy + meta.json as a third input option (alongside RAW and PNG bootstrap).
  - Scope is developer/testing support only (not a pipeline stage; no changes to run-folder layout, manifest.json, or viewer behavior).
  - Done when: developers can reliably generate small RAW crops (e.g., 256–1024 px) to speed up iteration, isolate artifacts, and perform reproducible A/B comparisons.

- **M7 — Scene-pack runner + consolidated report**
  - Add a small “scene-pack” runner that can process a folder of inputs (PNG, RAW/DNG, and crop.npy + meta.json) under two configurations (baseline vs candidate) and produce a consolidated metrics report.
  - Input: directory of images + configuration for baseline and candidate (e.g., two configs or two sets of CLI `--set` overrides); allow skipping invalid inputs when desired.
  - Output: a JSON report file (e.g., `reports/scene_pack_<name>.json`) with one row per image per config, including key metrics (from existing metrics artifacts), runtime, and links/paths to the corresponding run folders. Optionally include a simple “top regressions” section (e.g., worst ΔPSNR / highest halo proxy score).
  - No viewer changes required; reuse the existing run layout, metrics/diagnostics, and compare bundles.
  - Done when: you can point the tool at a folder of scenes, run baseline vs candidate, and get a single report that makes it easy to spot wins/regressions and jump into specific runs for inspection.
  
Notes:
- v0.2 can be developed using PNG-bootstrap runs; M5/M6 adds RAW/DNG input support to make diagnostics and A/B comparisons more representative on real sensor data.

## v0.3 — Viewer diagnostics & tone/DRC refinement
- Surface existing metrics/diagnostics directly in the static viewer UI to make per-stage behavior easier to inspect.
- Use the improved diagnostics flow to refine classical tone/DRC (e.g., `reinhard` / `filmic`) for better highlight handling and local contrast.
- Add lightweight HDR-readiness hooks in `tone` (scene-referred options, rolloff controls, simple DRC profiles) without changing run layout or `manifest.json` schema.

### Milestones (v0.3)
- **M1 — Diagnostics surfaced in viewer (non-breaking)**
  - Add lightweight viewer UI to surface metrics/diagnostics already emitted in v0.2 (e.g., per-stage numeric metrics panel, and simple toggles to view diagnostic PNGs such as diffs or false-color where present).
  - Use only existing JSON/PNG artifacts under `stages/<nn>_<name>/extra/`; do not change run layout, `manifest.json` schema, or diagnostics file formats.
  - Work in both single-run and compare modes; when metrics/diagnostics are missing for a stage or side, fall back gracefully (e.g., show “N/A”).
  - Done when: the viewer can show basic numeric metrics for the current stage and, where available, let the user view at least one diagnostic overlay per side—without regressing any existing viewer behavior.

- **M2 — Classical tone/DRC refinement**
  - Refine `tone` / DRC baselines (e.g., `tone.method: reinhard|filmic`) using v0.2 compare mode + metrics/diagnostics + the new viewer panels, targeting better highlight handling and local contrast without obvious artifacts.
  - Keep outputs deterministic; tune params (e.g., shoulder strength, mid-tone contrast) and record them cleanly in `debug.json`.
  - Done when: A/B comparisons on a small scene pack show consistent improvement (metrics + visual inspection) with no regressions in dark regions or skin tones.

- **M3 — HDR-readiness hooks**
  - Add HDR-readiness hooks in `tone`: scene-referred options, highlight rolloff controls, and a simple “DRC profile” parameter set (e.g., `tone.profile: standard|hdr_soft|hdr_strong`).
  - Keep the pipeline single-frame and deterministic; no multi-frame or AI here—just parameterized classical tone/DRC that can later be driven by multi-frame/AI logic.
  - Keep run-folder layout and `manifest.json` schema unchanged; all changes must be measurable via A/B comparison and existing diagnostics.

## v0.4 — AI-DRC option (contained)
- Introduce `tone.method: ai_drc` as a contained option that predicts a curve/LUT/coeff set and then applies it with the existing fast tone-mapping machinery (no full pixel-synthesis models at first).
- Keep AI-DRC deterministic and debuggable: expose predicted curves/LUTs as artifacts, and make it easy to A/B against classical tone methods using compare bundles.
- Optionally add an ONNX export hook so the learned DRC module can be used in other environments, without changing the mini-ISP run layout or viewer schema.

## v0.5+ — Multi-frame wrappers
- Add MFNR/HDR orchestration as **wrappers** around the existing single-frame stages (e.g., burst alignment + merge feeding into the same RAW→RGB→display pipeline).
- Keep single-frame stage contracts stable; multi-frame logic lives in composite/wrapper stages so existing runs and viewer behavior remain valid.
- Use the same compare/metrics/diagnostics infrastructure to evaluate multi-frame vs single-frame quality and performance.

## Ongoing — Method packs (non-breaking)
- Add new `*.method` options inside existing stages opportunistically (e.g., improved demosaic, denoise, tone, color_adjust), without changing run layout or manifest schema.
- Each new method must be:
  - deterministic and minimal-deps,
  - test-covered (pytest),
  - and evaluable via A/B compare + metrics/diagnostics.
- Method-pack changes do **not** define new roadmap versions; they build on the stable backbone defined by v0.1–v0.5.
