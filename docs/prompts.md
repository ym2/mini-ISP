# Prompts

This file logs the **final milestone prompts** (and any important **patch prompts**) used with **Codex (or any coding agent)**.

---

## M1 — Bootstrap runnable pipeline

### Final prompt
Implement a repo skeleton + a runnable pipeline runner that produces a run folder + `manifest.json` + a static viewer from a PNG input.  
Only `raw_norm` is real; all other stages are explicit stubs that still dump required artifacts (`preview.png`, `debug.json`, `timing_ms.json`, and `roi.png` when enabled).  
Follow `docs/artifacts_and_viewer.md`, `docs/stage_contracts.md`, `docs/pipeline.md`. Do not edit docs.  
Keep dependencies minimal (prefer numpy, PyYAML, and Pillow/imageio; avoid rawpy/torch/opencv).  
No internet downloads—if `data/sample.png` is missing, generate it deterministically (e.g., gradient + color bars) and save it under `data/`.  
End by giving one command that runs successfully and produces the expected `runs/<run_id>/` layout + viewer.

### Patch prompt
Please do a quick M1 sanity pass on `raw_norm` and patch only what’s necessary (do not edit docs).  
Check 1: `raw_norm` output must be a 2D `np.float32` Bayer mosaic with values roughly in `[0,1]` (clip if needed).  
Check 2: ensure `meta.cfa_pattern` is set (default `"RGGB"` for PNG bootstrap) and propagated consistently (stage debug + manifest input).  
If either check fails, make the minimal code changes and add minimal debug fields in `00_raw_norm/debug.json` to confirm: `dtype`, `shape`, `min`, `max` (keep existing `p01/p99`).  
No new dependencies; no change to run folder schema.  
End by telling me the exact command to run and which file(s) to inspect to verify both checks.

---

## M2 — Stage interface + artifacts helpers + smoke test

### Final prompt
Milestone M2 (v0.1): Stage interface + artifacts helpers + smoke test.  
Implement M2 per `docs/roadmap.md`: introduce a formal stage interface (typed I/O + `run()` contract) and small artifacts helper utilities (write `preview/roi/debug/timing` consistently).  
Add a basic end-to-end smoke test that passes with `pytest -q`.  
Constraints: keep current runtime behavior and run-folder layout unchanged; do not edit docs; keep dependencies minimal (no new heavy deps).  
Do not change `manifest.json` schema or viewer paths.  
Ensure every v0.1 stage (including stubs) emits required artifacts: `preview.png`, `debug.json`, `timing_ms.json`, and `roi.png` when ROI is enabled.  
Testing setup: `pytest` is not installed yet — add dev test dependency support (prefer `requirements-dev.txt`) without bloating runtime deps.  
Deliverables: end with (1) one command to install test deps, (2) one command to run tests, and (3) one command to run the pipeline.

---

## M3 — RAW-domain DPC + LSC

### Final prompt
Milestone M3 (v0.1): RAW-domain DPC + LSC.
Implement M3 per docs/roadmap.md: add real dpc and lsc stages operating on RAW_BAYER_F32.
DPC (dpc): implement median-of-neighbors on the RAW mosaic using a 3×3 window excluding the center pixel; for borders, either skip correction or use edge-clamped neighbors (choose one and keep deterministic). Add a tests-only defect injection helper that flips N known coordinates to 0 or 1. Report n_fixed in debug.json.
LSC (lsc): implement a deterministic radial gain map g(r)=1+k*(r/R)^2 (or equivalent monotonic radial falloff), capped by gain_cap, applied multiplicatively in RAW domain. Report gain stats (min/max/mean) in debug.json. Be explicit about output range handling: keep float32; if clipping to [0,1] is used, state it in debug.
Constraints: keep run-folder layout + manifest.json schema unchanged; do not edit docs; minimal deps; keep artifacts consistent; pytest -q must pass. Do not add new pipeline modes (stay within existing classic). Keep PNG bootstrap working (M1 still passes).
Tests: (1) inject defects then verify DPC fixes exactly those pixels (and n_fixed==N), (2) verify LSC gain never exceeds gain_cap and is radially symmetric.
Deliverables: end with (1) one command to install dev/test deps, (2) one command to run pytest -q, and (3) one command to run the pipeline (classic mode).
---

## M4 — WB + demosaic

### Final prompt
Milestone M4 (v0.1): WB + demosaic (RAW→linear RGB).
Implement wb_gains and demosaic per docs/roadmap.md and docs/stage_contracts.md.
wb_gains: apply gains in RAW mosaic domain for meta.cfa_pattern="RGGB" (R at [0,0], Gr at [0,1], Gb at [1,0], B at [1,1]; apply the G gain to both green sites). Set meta.wb_gains=[r,g,b] and meta.wb_applied=true. Required debug fields: wb_gains, wb_applied (and min/max after WB if already reporting stats).
demosaic: convert RAW_BAYER_F32 → RGB_LINEAR_F32 with demosaic.method: bilinear|malvar (v0.1 default = bilinear; implement malvar if feasible, otherwise wire the enum and raise a clear “not implemented” error when selected). For bilinear, define border handling as edge-clamp (replicate nearest valid pixel) for deterministic behavior. Output must be float32 with shape H×W×3. Be explicit about range handling (no-clip or clip; if clip, record clip_applied=true/false and clip range in debug). Required debug fields: method, clip_applied (plus any existing min/max/p01/p99 stats).
Tests: verify WB on a small synthetic mosaic (known values → expected scaled values). Verify demosaic output shape/dtype for both methods; for bilinear, also sanity-check values are within a reasonable range and border behavior is deterministic. Ensure required debug fields are present.
Constraints: keep run-folder layout + manifest.json schema unchanged; do not edit docs; minimal deps; pytest -q must pass.
Deliverables: end with commands to install dev/test deps, run pytest -q, and run the pipeline (classic mode).

---

## M5 — Denoise baseline + JDD composite

### Final prompt
Milestone M5 (v0.1): Denoise baseline + JDD composite.
Implement an RGB denoise stage denoise operating on RGB_LINEAR_F32 with denoise.method: gaussian|box (v0.1 default = gaussian). Implement in NumPy only (no NLMeans / no new heavy deps). Define border handling as edge-clamp (replicate nearest pixel) for deterministic results. Keep output float32. Defaults (v0.1): gaussian uses sigma=1.0, ksize=5 (odd); box uses ksize=3 (odd). Be explicit about range handling (no-clip or clip; if clip, record clip_applied and clip_range in debug). Required debug fields: method, params (e.g., ksize/sigma), clip_applied (and existing stats if present).
Add a jdd_raw2rgb composite stage used by pipeline_mode: jdd that replaces demosaic + denoise. The composite must preserve I/O contracts: input RAW_BAYER_F32, output RGB_LINEAR_F32, and must reuse the existing demosaic + denoise implementations internally for jdd.method: wrapper (v0.1). Ensure stage artifacts remain consistent (preview/debug/timing/roi).
Constraints: keep run-folder layout + manifest.json schema unchanged; do not edit docs; minimal deps; pytest -q must pass; keep PNG bootstrap working.
Tests:
1.Denoise reduces noise deterministically: create a clean synthetic RGB image, add seeded Gaussian noise (fixed seed), run denoise (default params), and assert variance (or MSE vs clean) decreases; also assert shape/dtype unchanged and values remain finite.
2.pipeline_mode=jdd runs end-to-end and produces the required artifacts for every stage (at least: preview.png, debug.json, timing_ms.json, and roi.png when ROI enabled), matching the artifact presence rules used in classic.
Deliverables: end with commands to (1) install dev/test deps, (2) run pytest -q, and (3) run the pipeline in both classic and jdd modes.
