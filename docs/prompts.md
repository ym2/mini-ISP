# Prompts

This file logs the **final milestone prompts** (and any important **patch prompts**) used with **Codex (or any coding agent)**.

---

## M1 — Bootstrap runnable pipeline

### Final prompt
Implement a repo skeleton + a runnable pipeline runner that produces a run folder + `manifest.json` + a static viewer from a PNG input.

- Only `raw_norm` is real; all other stages are explicit stubs that still dump required artifacts (`preview.png`, `debug.json`, `timing_ms.json`, and `roi.png` when enabled).
- Follow `docs/artifacts_and_viewer.md`, `docs/stage_contracts.md`, `docs/pipeline.md`. Do not edit docs.
- Keep dependencies minimal (prefer numpy, PyYAML, and Pillow/imageio; avoid rawpy/torch/opencv).
- No internet downloads—if `data/sample.png` is missing, generate it deterministically (e.g., gradient + color bars) and save it under `data/`.
- End by giving one command that runs successfully and produces the expected `runs/<run_id>/` layout + viewer.

### Patch prompt
Please do a quick M1 sanity pass on `raw_norm` and patch only what’s necessary (do not edit docs).

- Check 1: `raw_norm` output must be a 2D `np.float32` Bayer mosaic with values roughly in `[0,1]` (clip if needed).
- Check 2: ensure `meta.cfa_pattern` is set (default `"RGGB"` for PNG bootstrap) and propagated consistently (stage debug + manifest input).
- If either check fails, make the minimal code changes and add minimal debug fields in `00_raw_norm/debug.json` to confirm: `dtype`, `shape`, `min`, `max` (keep existing `p01/p99`).
- No new dependencies; no change to run folder schema.
- End by telling me the exact command to run and which file(s) to inspect to verify both checks.

---

## M2 — Stage interface + artifacts helpers + smoke test

### Final prompt
Milestone M2 (v0.1): Stage interface + artifacts helpers + smoke test.

- Implement M2 per `docs/roadmap.md`: introduce a formal stage interface (typed I/O + `run()` contract) and small artifacts helper utilities (write `preview/roi/debug/timing` consistently).
- Add a basic end-to-end smoke test that passes with `pytest -q`.
- Constraints: keep current runtime behavior and run-folder layout unchanged; do not edit docs; keep dependencies minimal (no new heavy deps).
- Do not change `manifest.json` schema or viewer paths.
- Ensure every v0.1 stage (including stubs) emits required artifacts: `preview.png`, `debug.json`, `timing_ms.json`, and `roi.png` when ROI is enabled.
- Testing setup: `pytest` is not installed yet — add dev test dependency support (prefer `requirements-dev.txt`) without bloating runtime deps.
- Deliverables: end with (1) one command to install test deps, (2) one command to run tests, and (3) one command to run the pipeline.

---

## M3 — RAW-domain DPC + LSC

### Final prompt
Milestone M3 (v0.1): RAW-domain DPC + LSC.

- Implement M3 per docs/roadmap.md: add real dpc and lsc stages operating on RAW_BAYER_F32.
- DPC (dpc): implement median-of-neighbors on the RAW mosaic using a 3×3 window excluding the center pixel; for borders, either skip correction or use edge-clamped neighbors (choose one and keep deterministic). Add a tests-only defect injection helper that flips N known coordinates to 0 or 1. Report n_fixed in debug.json.
- LSC (lsc): implement a deterministic radial gain map g(r)=1+k*(r/R)^2 (or equivalent monotonic radial falloff), capped by gain_cap, applied multiplicatively in RAW domain. Report gain stats (min/max/mean) in debug.json. Be explicit about output range handling: keep float32; if clipping to [0,1] is used, state it in debug.
- Constraints: keep run-folder layout + manifest.json schema unchanged; do not edit docs; minimal deps; keep artifacts consistent; pytest -q must pass. Do not add new pipeline modes (stay within existing classic). Keep PNG bootstrap working (M1 still passes).
- Tests: (1) inject defects then verify DPC fixes exactly those pixels (and n_fixed==N), (2) verify LSC gain never exceeds gain_cap and is radially symmetric.
- Deliverables: end with (1) one command to install dev/test deps, (2) one command to run pytest -q, and (3) one command to run the pipeline (classic mode).

---

## M4 — WB + demosaic

### Final prompt
Milestone M4 (v0.1): WB + demosaic (RAW→linear RGB).

- Implement wb_gains and demosaic per docs/roadmap.md and docs/stage_contracts.md.
- wb_gains: apply gains in RAW mosaic domain for meta.cfa_pattern="RGGB" (R at [0,0], Gr at [0,1], Gb at [1,0], B at [1,1]; apply the G gain to both green sites). Set meta.wb_gains=[r,g,b] and meta.wb_applied=true. Required debug fields: wb_gains, wb_applied (and min/max after WB if already reporting stats).
- demosaic: convert RAW_BAYER_F32 → RGB_LINEAR_F32 with demosaic.method: bilinear|malvar (v0.1 default = bilinear; implement malvar if feasible, otherwise wire the enum and raise a clear “not implemented” error when selected). For bilinear, define border handling as edge-clamp (replicate nearest valid pixel) for deterministic behavior. Output must be float32 with shape H×W×3. Be explicit about range handling (no-clip or clip; if clip, record clip_applied=true/false and clip range in debug). Required debug fields: method, clip_applied (plus any existing min/max/p01/p99 stats).
- Tests: verify WB on a small synthetic mosaic (known values → expected scaled values). Verify demosaic output shape/dtype for both methods; for bilinear, also sanity-check values are within a reasonable range and border behavior is deterministic. Ensure required debug fields are present.
- Constraints: keep run-folder layout + manifest.json schema unchanged; do not edit docs; minimal deps; pytest -q must pass.
- Deliverables: end with commands to install dev/test deps, run pytest -q, and run the pipeline (classic mode).

---

## M5 — Denoise baseline + JDD composite

### Final prompt
Milestone M5 (v0.1): Denoise baseline + JDD composite.

- Implement an RGB denoise stage denoise operating on RGB_LINEAR_F32 with denoise.method: gaussian|box (v0.1 default = gaussian). Implement in NumPy only (no NLMeans / no new heavy deps). Define border handling as edge-clamp (replicate nearest pixel) for deterministic results. Keep output float32. Defaults (v0.1): gaussian uses sigma=1.0, ksize=5 (odd); box uses ksize=3 (odd). Be explicit about range handling (no-clip or clip; if clip, record clip_applied and clip_range in debug). Required debug fields: method, params (e.g., ksize/sigma), clip_applied (and existing stats if present).
- Add a jdd_raw2rgb composite stage used by pipeline_mode: jdd that replaces demosaic + denoise. The composite must preserve I/O contracts: input RAW_BAYER_F32, output RGB_LINEAR_F32, and must reuse the existing demosaic + denoise implementations internally for jdd.method: wrapper (v0.1). Ensure stage artifacts remain consistent (preview/debug/timing/roi).
- Constraints: keep run-folder layout + manifest.json schema unchanged; do not edit docs; minimal deps; pytest -q must pass; keep PNG bootstrap working.
- Tests:
  - Denoise reduces noise deterministically: create a clean synthetic RGB image, add seeded Gaussian noise (fixed seed), run denoise (default params), and assert variance (or MSE vs clean) decreases; also assert shape/dtype unchanged and values remain finite.
  - pipeline_mode=jdd runs end-to-end and produces the required artifacts for every stage (at least: preview.png, debug.json, timing_ms.json, and roi.png when ROI enabled), matching the artifact presence rules used in classic.
- Deliverables: end with commands to (1) install dev/test deps, (2) run pytest -q, and (3) run the pipeline in both classic and jdd modes.

---

## M6 — CCM + 3A stats

### Final prompt
Milestone M6 (v0.1): CCM + 3A stats.

Implement ccm (3×3 color correction matrix) and stats_3a per docs/stage_contracts.md.

CCM (ccm):
- Operates on RGB_LINEAR_F32 → RGB_LINEAR_F32.
- Support ccm.mode: identity|manual|profile (v0.1 default = identity).
- identity: use I3.
- manual: read 3×3 from config.
- profile (v0.1 stub): fall back to identity and emit a warning (do not break the pipeline).
- Record meta.ccm (always the 3×3 used, even identity) and meta.ccm_mode.
- Be explicit about range handling: either no-clip, or clip to a configured range; if clip, record clip_applied and clip_range in debug.
- Required debug fields: mode, matrix (3×3), clip_applied (+ clip_range if used), and basic stats (min/max/p01/p99).

3A stats (stats_3a):
- Non-invasive: input/output pixels must be identical (RGB_LINEAR_F32 → RGB_LINEAR_F32).
- Compute AE/AWB/AF-style measurements and store them in debug.json.metrics.stats_3a (and in meta.stats_3a if meta plumbing already supports it without refactors).
- AE stats:
  - Luma definition: Y = 0.2126R + 0.7152G + 0.0722B (Rec.709).
  - Histogram: n_bins=64, range [0, 1] (clip luma to [0,1] for hist only).
  - Report mean, p01, p99, and clip_pct (percent of pixels with luma >= 1.0 or <= 0.0, based on unclipped values).
- AWB stats:
  - Gray-world channel means over the same region: mean_r, mean_g, mean_b (+ optionally ratios to mean_g).
- AF stats:
  - Focus metric: Tenengrad on ROI (use configured ROI when enabled; otherwise use a deterministic center crop, e.g., 25% area).
  - Define Tenengrad as mean of squared Sobel gradient magnitude on luma (deterministic border handling = edge-clamp).

Constraints: keep run-folder layout + manifest.json schema unchanged; do not edit docs; minimal deps; pytest -q must pass; keep PNG bootstrap working.

Tests:
- CCM: identity mode does not change pixels; manual mode matches expected output on a small synthetic RGB array (float32).
- stats_3a: stage output pixels are bitwise-equal to input; required keys exist with sensible ranges/shapes (hist sums correctly; focus metric is finite).

Deliverables: end with commands to (1) install dev/test deps, (2) run pytest -q, and (3) run the pipeline (classic mode).

---

## M7 — Tone + color adjust + sharpen + OETF

### Final prompt
Milestone M7 (v0.1): Tone + color adjust + sharpen + OETF.

Implement tone, color_adjust, sharpen, and oetf_encode per docs/stage_contracts.md.

Global policy (v0.1):
- Keep intermediate stages in float32 and allow values outside [0,1] (no implicit clipping), unless explicitly configured.
- Only oetf_encode must clip to [0,1] before encoding to display output. If any earlier stage applies clipping, it must be opt-in and recorded in debug.

tone:
- Add tone.method: reinhard|filmic (v0.1 default = reinhard).
- Operate on RGB_LINEAR_F32 → RGB_LINEAR_F32, keep float32.
- Record in debug: method, params, and range stats; if clipping is applied (opt-in), record clip_applied and clip_range.

color_adjust:
- Add color_adjust.method: identity|chroma_scale_lrgb (v0.1 default = identity).
- chroma_scale_lrgb is the v0.1 baseline (minimal-deps, deterministic). Future methods (v0.2+) will add: oklab_chroma_scale and ictcp_chroma_scale.
- identity: no-op (must be bitwise-equal output).
- chroma_scale_lrgb: neutral-axis chroma scaling in linear RGB:
- Y = 0.2126R + 0.7152G + 0.0722B
- rgb' = Y + sat_scale * (rgb - Y) with sat_scale (default 1.0).
- Operate on RGB_LINEAR_F32 → RGB_LINEAR_F32, keep float32.
- Record in debug: method, sat_scale, and range stats; no clipping by default.

sharpen:
- Add sharpen.method: unsharp_mask (v0.1 default = unsharp_mask).
- Implement unsharp mask with deterministic blur and borders:
- blur = Gaussian (default sigma=1.0, kernel derived from sigma) and border handling = edge-clamp
- params: amount (default 0.5), threshold (default 0.0), sigma (default 1.0)
- Operate on RGB_LINEAR_F32 → RGB_LINEAR_F32, keep float32.
- Record in debug: method, sigma, amount, threshold, and range stats; no clipping by default.

oetf_encode:
- Add oetf_encode.oetf: srgb (v0.1 default = srgb).
- Apply the selected OETF (v0.1: sRGB OETF, standard piecewise) and output RGB_DISPLAY_U8.
- Before OETF, clip to [0,1] (required in v0.1). Record clip_applied=true and clip_range=[0,1].
- Save final output image in final/ and ensure stage artifacts are dumped like others.
- Record in debug: oetf (srgb), output dtype/bit depth (uint8, 8-bit), and clip info.

Constraints: keep run-folder layout + manifest.json schema unchanged; do not edit docs; minimal deps; pytest -q must pass; PNG bootstrap still works.

Tests:
1. tone: both methods preserve shape/dtype; reinhard on a known input produces predictable compression (e.g., monotonic, reduced highlights).
2. color_adjust: identity is a strict no-op (bitwise-equal); chroma_scale_lrgb changes chroma as expected on a synthetic RGB triplet while preserving luma.
3. sharpen: unsharp_mask increases local contrast on a synthetic step edge (e.g., higher gradient magnitude) without changing shape/dtype.
4. oetf_encode: oetf=srgb returns uint8 in [0,255], and clipping is recorded.

Deliverables: end with commands to (1) install dev/test deps, (2) run pytest -q, and (3) run the pipeline in classic mode.
