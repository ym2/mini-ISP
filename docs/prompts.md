# Prompts

This file logs the **final milestone prompts** (and any important **patch prompts**) used with **Codex (or any coding agent)**.

---

## v0.1-M1 — Bootstrap runnable pipeline

### Final prompt
Implement a repo skeleton + a runnable pipeline runner that produces a run folder + `manifest.json` + a static viewer from a PNG input.

- Only `raw_norm` is real; all other stages are explicit stubs that still dump required artifacts (`preview.png`, `debug.json`, `timing_ms.json`, and `roi.png` when enabled).
- Follow `docs/artifacts_and_viewer.md`, `docs/stage_contracts.md`, `docs/pipeline.md`. Do not edit docs.
- Keep dependencies minimal (prefer numpy, PyYAML, and Pillow/imageio; avoid rawpy/torch/opencv).
- No internet downloads—if `data/sample.png` is missing, generate it deterministically (e.g., gradient + color bars) and save it under `data/`.
- End by giving one command that runs successfully and produces the expected `runs/<run_id>/` layout + viewer.

### Patch prompt 1
Please do a quick M1 sanity pass on `raw_norm` and patch only what’s necessary (do not edit docs).

- Check 1: `raw_norm` output must be a 2D `np.float32` Bayer mosaic with values roughly in `[0,1]` (clip if needed).
- Check 2: ensure `meta.cfa_pattern` is set (default `"RGGB"` for PNG bootstrap) and propagated consistently (stage debug + manifest input).
- If either check fails, make the minimal code changes and add minimal debug fields in `00_raw_norm/debug.json` to confirm: `dtype`, `shape`, `min`, `max` (keep existing `p01/p99`).
- No new dependencies; no change to run folder schema.
- End by telling me the exact command to run and which file(s) to inspect to verify both checks.

---

## v0.1-M2 — Stage interface + artifacts helpers + smoke test

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

## v0.1-M3 — RAW-domain DPC + LSC

### Final prompt
Milestone M3 (v0.1): RAW-domain DPC + LSC.

- Implement M3 per docs/roadmap.md: add real dpc and lsc stages operating on RAW_BAYER_F32.
- DPC (dpc): implement median-of-neighbors on the RAW mosaic using a 3×3 window excluding the center pixel; for borders, either skip correction or use edge-clamped neighbors (choose one and keep deterministic). Add a tests-only defect injection helper that flips N known coordinates to 0 or 1. Report n_fixed in debug.json.
- LSC (lsc): implement a deterministic radial gain map g(r)=1+k*(r/R)^2 (or equivalent monotonic radial falloff), capped by gain_cap, applied multiplicatively in RAW domain. Report gain stats (min/max/mean) in debug.json. Be explicit about output range handling: keep float32; if clipping to [0,1] is used, state it in debug.
- Constraints: keep run-folder layout + manifest.json schema unchanged; do not edit docs; minimal deps; keep artifacts consistent; pytest -q must pass. Do not add new pipeline modes (stay within existing classic). Keep PNG bootstrap working (M1 still passes).
- Tests: (1) inject defects then verify DPC fixes exactly those pixels (and n_fixed==N), (2) verify LSC gain never exceeds gain_cap and is radially symmetric.
- Deliverables: end with (1) one command to install dev/test deps, (2) one command to run pytest -q, and (3) one command to run the pipeline (classic mode).

### Patch prompt 1 — CFA-aware DPC (same-CFA neighbors only)
Patch M3 DPC: DPC runs on a Bayer mosaic, so any replacement value must be computed from **same-CFA sites only** (no cross-color borrowing). Make this invariant the default (no legacy mode).

- Change dpc to compute the replacement as the median of the 8 same-plane neighbors at ±2 offsets: (±2,0), (0,±2), (±2,±2). For borders: ignore out-of-bounds neighbors (median of the available same-plane neighbors only; no edge clamping to other CFA planes).
- Keep the existing config interface unchanged.
- Record in debug.json: `metrics.neighbor_policy="same_cfa_only"`, `metrics.neighbor_stat="median"`, and the threshold used.
- Tests:
  - On a synthetic mosaic with different constant values per CFA plane (R/G1/G2/B), DPC must make zero changes (n_fixed==0) across RGGB/BGGR/GRBG/GBRG.
  - Inject a single defect at each CFA site (R/G1/G2/B) and assert DPC fixes exactly that pixel using same-plane neighbors only, across RGGB/BGGR/GRBG/GBRG.

### Patch prompt 2 — Explicit LSC on/off toggle
Add support for `stages.lsc.enabled: true|false` (default true).

- If `enabled=false`, bypass `lsc` (input RAW mosaic unchanged) but still emit a normal stage folder and `debug.json` noting it was skipped so the viewer/run layout stays consistent.
- Keep existing implicit-disable options working (`stages.lsc.k=0`, `stages.lsc.gain_cap=1`).
- Implementation constraint: `pipeline_mode` stage lists are fixed, so the toggle must be handled by the runner (or stage wrapper), not by adding logic only inside `stage_lsc`.
- Debug: when skipped, ensure `debug.json.params.enabled=false` and `debug.json.metrics.skipped=true` (or equivalent).
- Tests:
  - Integration-ish: run the pipeline on PNG bootstrap with `--set stages.lsc.enabled=false` and assert `02_lsc` preview matches `01_dpc` preview exactly and debug reports it was skipped.

---

## v0.1-M4 — WB + demosaic

### Final prompt
Milestone M4 (v0.1): WB + demosaic (RAW→linear RGB).

- Implement wb_gains and demosaic per docs/roadmap.md and docs/stage_contracts.md.
- wb_gains: apply gains in RAW mosaic domain for meta.cfa_pattern="RGGB" (R at [0,0], Gr at [0,1], Gb at [1,0], B at [1,1]; apply the G gain to both green sites). Set meta.wb_gains=[r,g,b] and meta.wb_applied=true. Required debug fields: wb_gains, wb_applied (and min/max after WB if already reporting stats).
- demosaic: convert RAW_BAYER_F32 → RGB_LINEAR_F32 with demosaic.method: bilinear|malvar (v0.1 default = bilinear; implement malvar if feasible, otherwise wire the enum and raise a clear “not implemented” error when selected). For bilinear, define border handling as edge-clamp (replicate nearest valid pixel) for deterministic behavior. Output must be float32 with shape H×W×3. Be explicit about range handling (no-clip or clip; if clip, record clip_applied=true/false and clip range in debug). Required debug fields: method, clip_applied (plus any existing min/max/p01/p99 stats).
- Tests: verify WB on a small synthetic mosaic (known values → expected scaled values). Verify demosaic output shape/dtype for both methods; for bilinear, also sanity-check values are within a reasonable range and border behavior is deterministic. Ensure required debug fields are present.
- Constraints: keep run-folder layout + manifest.json schema unchanged; do not edit docs; minimal deps; pytest -q must pass.
- Deliverables: end with commands to install dev/test deps, run pytest -q, and run the pipeline (classic mode).

---

## v0.1-M5 — Denoise baseline + JDD composite

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

## v0.1-M6 — CCM + 3A stats

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

### Patch prompt 1 — CCM chain mode (opt-in, no stage split)
Add a new CCM mode that makes the camera/working-space transform explicit, while keeping the stage name `ccm` and pipeline stage order unchanged.

Implementation:
- Add `stages.ccm.mode: chain`.
- `chain` uses two 3×3 matrices (float32), sourced from config keys:
  1) `cam_to_xyz_matrix` (camera RGB → XYZ D65). Internally refer to this as `M_cam_to_xyz`.
  2) `xyz_to_working_matrix` (XYZ D65 → working linear RGB). Internally refer to this as `M_xyz_to_working`.
     If `xyz_to_working_matrix` is not provided, default to a built-in constant matrix for XYZ(D65) → linear sRGB (D65).
- Multiply convention (record verbatim): pixels are row vectors; apply a single 3×3 as `out = in @ M.T`.
- With that convention, the sequential chain is:
  - `xyz = in @ M_cam_to_xyz.T`
  - `out = xyz @ M_xyz_to_working.T`
- Compute and record:
  - `effective_matrix = M_xyz_to_working @ M_cam_to_xyz`
- Apply `effective_matrix` using the existing CCM multiply path so the output matches the sequential chain above:
  - `out = in @ effective_matrix.T`
- Keep existing CCM options (e.g., `clip`) working unchanged for `mode=chain`.

Config + fallback policy:
- `stages.ccm.cam_to_xyz_matrix` (3×3 list) is required for `mode=chain`.
- `stages.ccm.xyz_to_working_matrix` (3×3 list) is optional; when missing, use the built-in XYZ(D65) → linear sRGB matrix.
- If `cam_to_xyz_matrix` is missing in `mode=chain`: fall back to the identity matrix for `cam_to_xyz_matrix` (I3) and set `cam_to_xyz_source="identity_fallback"` (do not guess).
- Still record all matrices/debug fields deterministically in this fallback case (including `xyz_to_working_matrix` and `effective_matrix`).
- Do not change run layout, `manifest.json` schema, or viewer paths/assets in this patch.

Debug contract (stage debug.json at stage root):
- Under `params` record:
  - `mode: "chain"`
  - `working_space: "lin_srgb_d65"` (fixed for now)
  - `mul_convention: "out = in @ M.T"`
  - `cam_to_xyz_source: "config"|"identity_fallback"`
  - `xyz_to_working_source: "constant_xyz_to_lin_srgb_d65"|"config"`
  - `cam_to_xyz_matrix` (3×3 list)
  - `xyz_to_working_matrix` (3×3 list)
  - `effective_matrix` (3×3 list)
- For backward compatibility, always populate:
  - `meta.ccm_mode = "chain"`
  - `meta.ccm = effective_matrix` (3×3 list)
  - This must still hold when `cam_to_xyz_source="identity_fallback"`.

Tests (pytest):
- Equivalence (math + convention): for fixed `cam_to_xyz_matrix` and `xyz_to_working_matrix`, compute `effective_matrix = xyz_to_working_matrix @ cam_to_xyz_matrix` in the test and assert:
  - `stage_ccm(mode=chain)` output ≈ `stage_ccm(mode=manual, matrix=effective_matrix)` output (within epsilon).
- Identity: if both matrices are identity, output must be unchanged.
- Ensure dtype/shape preserved; all output values finite.
- Fallback: in `mode=chain` with missing `cam_to_xyz_matrix`, assert `cam_to_xyz_source="identity_fallback"` and that `meta.ccm_mode="chain"` / `meta.ccm` are still populated deterministically.

Deliverables:
- One command to run tests: `pytest -q`
- One example config snippet:
```yaml
stages:
  ccm:
    mode: chain
    cam_to_xyz_matrix:
      - [0.70, 0.20, 0.10]
      - [0.10, 0.90, 0.00]
      - [0.00, 0.10, 0.90]
```

---

## v0.1-M7 — Tone + color adjust + sharpen + OETF

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

---

## v0.1-M8 — drc_plus curated composite

### Final prompt
Milestone M8 (v0.1): drc_plus curated composite.

- Implement drc_plus_color to replace tone + color_adjust in pipeline_mode: drc_plus per docs/pipeline.md and docs/stage_contracts.md. The composite must reuse the existing tone and color_adjust code paths internally (wrapper/composition: call tone then color_adjust) and preserve I/O contracts: input RGB_LINEAR_F32, output RGB_LINEAR_F32. Add drc_plus_color.method: wrapper (v0.1 default) and pass through nested params for tone and color_adjust.
- Debug requirements: debug.json must include stage, method, expands_to: ["tone","color_adjust"], nested summaries (tone.method/params, color_adjust.method/params), and (optional) timing_breakdown_ms for the two sub-steps.
- Constraints: keep run-folder layout + manifest.json schema unchanged; do not edit docs; minimal deps; pytest -q must pass; PNG bootstrap working.
- Tests: (1) pipeline_mode=drc_plus runs end-to-end and produces required artifacts for every stage; (2) drc_plus_color output matches sequential tone→color_adjust on a synthetic RGB_LINEAR_F32 input (use tight np.allclose).
- Deliverables: end with commands to run tests and run the pipeline in drc_plus mode.

---

## v0.2-M1 — Compare bundle + viewer A/B compare

### Final prompt
Milestone v0.2-M1: Compare bundle + viewer A/B compare (foundation).

Implement a backward-compatible compare mode for the existing static viewer. Add support for loading either:
(1) a single run manifest.json (current behavior), OR
(2) a compare bundle JSON that points to two run folders (A and B), each containing manifest.json.

Compare bundle spec (must implement exactly):
{
  "schema_version": "0.1",
  "title": "...",
  "created_utc": "...",
  "a": { "label": "...", "run_dir": "runs/<idA>" },
  "b": { "label": "...", "run_dir": "runs/<idB>" },
  "notes": "...",
  "stage_match": { "primary": "index", "fallback": "name" }
}
Viewer behavior:
- If URL has ?manifest=PATH, load single manifest (unchanged behavior).
- If URL has ?compare=PATH, load compare bundle, then load A and B manifests.
- Show side-by-side previews (A and B) with synchronized stage selection.
- Show per-stage timing for A and B and a delta (B - A) if available.
- Stage matching: primarily by stage.index; if unavailable or mismatched, fallback to stage.name.
- If a stage is missing on one side after matching, show the available side and display N/A for the missing preview/timing; delta is N/A.
- Compare bundle path in examples: serve it under runs/compare/compare.json (or similar) and pass a server-relative path in ?compare=.
- Constraints: keep run-folder layout and manifest.json schema unchanged (do not edit manifest.json); no heavy deps / no build tooling (keep viewer static: index.html/app.js/styles.css); do not edit docs; preserve existing single-run behavior (?manifest=) exactly.

Tests:
- Add a minimal test that validates compare bundle parsing and stage matching logic in isolation (pure Python or pure JS test—pick the lightest approach consistent with current repo).
- pytest -q must pass.

Deliverables:
- End with (1) one command to run tests, and (2) exact commands to start a local server and open:
- single-run mode, and
- compare mode with a sample compare.json.

### Patch prompt 1 — CLI helper to generate compare bundles
Task:
- add `python -m mini_isp.compare` to generate a compare bundle JSON deterministically

CLI flags:
- --a RUN_DIR_A
- --b RUN_DIR_B
- --out OUT_PATH
- --label-a LABEL_A
- --label-b LABEL_B
- --notes NOTES            (optional; default "")
- --title TITLE            (optional; default "mini-ISP compare")
- --created-utc CREATED_UTC (optional; if omitted, use current UTC ISO8601)

Validation:
- require RUN_DIR_A/manifest.json exists; otherwise exit with a clear error
- require RUN_DIR_B/manifest.json exists; otherwise exit with a clear error
- create parent directories for OUT_PATH if needed

Output schema (must match exactly):
{
  "schema_version": "0.1",
  "title": "...",
  "created_utc": "...",
  "a": { "label": "...", "run_dir": "runs/<idA>" },
  "b": { "label": "...", "run_dir": "runs/<idB>" },
  "notes": "...",
  "stage_match": { "primary": "index", "fallback": "name" }
}

Constraints:
- do not change viewer behavior
- do not change manifest.json schema or any run-folder layout/paths
- do not edit docs
- no new heavy dependencies

Tests:
- add a minimal pytest that runs the CLI with a fixed --created-utc
- assert the generated JSON equals an expected dict (including stage_match)

Deliverables:
- (1) one command to run the CLI
- (2) one command to run `pytest -q`

### Patch prompt 2 — Example template + init copy
Task:
- add a tracked example file `examples/compare_bundle.example.json` that matches the compare bundle schema (with obvious placeholders)
- add a small convenience command that copies the example to a user-specified path

Behavior:
- `python -m mini_isp.compare --init OUT_PATH`
- copies `examples/compare_bundle.example.json` → OUT_PATH
- creates parent dirs for OUT_PATH if needed
- refuses to overwrite if OUT_PATH already exists (exit with a clear error)
- prints a one-line hint showing how to use the viewer with `?compare=...`

Constraints:
- do not change viewer behavior
- do not change manifest.json schema or any run-folder layout/paths
- do not edit docs
- no new heavy dependencies

Tests:
- add a minimal pytest that runs `python -m mini_isp.compare --init <tmp_path>/compare.json`
- assert the file is created and JSON loads with required top-level keys:
  schema_version, title, created_utc, a, b, notes, stage_match

Deliverables:
- (1) one command to run the init copy
- (2) one command to run pytest -q

---

## v0.2-M2 — Stage-diff metrics + diagnostics outputs

### Final prompt
v0.2-M2 — Stage-diff metrics + diagnostics outputs.
Implement optional metrics/diagnostics outputs without changing manifest.json or existing viewer paths. When enabled, write additive files under each stage folder at stages/<nn>_<name>/extra/.
Outputs (when enabled):
	•	extra/metrics.json: per-stage metrics computed from that stage’s preview.png (v0.2-M2 uses preview domain only). Include: min/max/p01/p99, clip_pct, luma mean, chroma means (define luma as Rec.709 on preview RGB; chroma can be simple (R−G, B−G) means).
	•	extra/diff_metrics.json: metrics between this stage’s preview.png and the previous stage in the emitted stage list (successive folders). Include L1, L2, and PSNR (optional if cheap).
	•	extra/diagnostics/ containing optional PNG/JSON diagnostics derived from preview images (proxy implementations are fine but must be deterministic):
	•	false-color map (chroma magnitude or chroma diff vs prev)
	•	zipper proxy (high-frequency chroma energy map/score)
	•	halo/ringing proxy (edge overshoot proxy map/score)
Constraints: keep run-folder layout and manifest.json schema unchanged; no heavy deps; no build tooling; no doc edits; metrics/diagnostics are optional/additive; deterministic outputs only.
Tests: run a small synthetic pipeline with metrics+diagnostics enabled; assert expected files exist and required keys exist; verify determinism by running twice and comparing JSON contents (and optionally file hashes); pytest -q passes.
Deliverables: one command to run tests; one command to run the pipeline with metrics/diagnostics enabled.

### Patch prompt 1 — CLI overrides for metrics/diagnostics (preview-only)
Task:
	•	add CLI overrides to python -m mini_isp.run so users can enable metrics/diagnostics without creating a YAML file
	•	keep existing --config behavior unchanged; overrides must merge on top of loaded/default config

New CLI flags:
	•	--enable-metrics (bool; default false)
	•	--enable-diagnostics (bool; default false)
	•	--metrics-target with choices preview|linear|both (default preview)
	•	--metrics-diff with choices off|l1|l2|psnr|all (default l1)
	•	--metrics-out with choices stage_root|extra (default extra)
	•	stage_root → stages/<nn>_<name>/metrics.json
	•	extra → stages/<nn>_<name>/extra/metrics.json

Behavior:
	•	if --enable-metrics is set, enable metrics in the resolved config
	•	if --enable-diagnostics is set, enable diagnostics in the resolved config
	•	these flags must work when --config is omitted
	•	preview-only for this milestone: if --metrics-target is linear or both, exit with a clear error like: “linear metrics not implemented yet; use –metrics-target preview”
	•	do not require /dev/null tricks; running with no config must succeed

Constraints:
	•	do not change manifest.json schema
	•	do not change run-folder layout or viewer paths
	•	do not edit docs
	•	no heavy deps; keep current deps
	•	keep outputs deterministic

Tests:
	•	add minimal pytest that runs runner via subprocess with:
	•	--enable-metrics --enable-diagnostics --metrics-target preview --metrics-diff l1
	•	assert at least one stage contains expected metrics/diagnostics output at the chosen location
	•	add one pytest that asserts --metrics-target linear fails with the expected error string
	•	pytest -q must pass

Deliverables:
	•	command to run tests
	•	command to run pipeline with overrides, e.g.:
	•	python -m mini_isp.run --input data/sample.png --out runs --pipeline_mode classic --name v02_m2_metrics --enable-metrics --enable-diagnostics

---

## v0.2-M3 — Denoise upgrade (beyond current Gaussian/box baseline)

### Final prompt
v0.2-M3 — Denoise upgrade (beyond current Gaussian/box baseline)

Implement a denoise improvement that measurably outperforms the current Gaussian/box blur baseline, without heavy deps. Add a new denoise method (keep existing methods intact); recommended: denoise.method: chroma_gaussian that applies stronger smoothing on chroma than luma using a simple, deterministic transform (e.g., Y + two chroma-like channels derived from RGB). Keep outputs deterministic and fast (NumPy only).

Requirements:
	•	I/O contracts unchanged: RGB_LINEAR_F32 → RGB_LINEAR_F32; dtype stays float32; no implicit clipping (if you choose to clip, it must be explicitly controlled and recorded).
	•	Config: keep existing denoise.method options; add chroma_gaussian with deterministic defaults (e.g., sigma_y=1.0, sigma_c=2.0, ksize=5, edge handling = clamp).
	•	Debug fields (in debug.json): method, params (sigmas/ksize/edge_mode), clip_applied, clip_range, and existing range stats (min/max/p01/p99).
	•	Keep run-folder layout, per-stage artifacts, and manifest.json schema unchanged; do not edit docs; no new heavy dependencies.

Tests (pytest):
	•	Deterministic synthetic test with fixed seed: create a clean RGB image; add seeded Gaussian noise; compare denoised outputs.
	•	Assert the new method yields lower MSE (and/or higher PSNR) vs the current gaussian baseline on the noisy input.
	•	Assert shape/dtype unchanged; values finite.

Deliverables:
	•	End with (1) one command to install dev/test deps, (2) one command to run pytest -q, and (3) one command to run the pipeline twice for A/B (baseline gaussian vs chroma_gaussian) so it can be compared later with v0.2-M1 compare mode.

### Patch prompt 1 — CLI --set overrides for config (no docs changes)
Task: add a lightweight CLI override mechanism to python -m mini_isp.run so I can override config values without writing a YAML file.
python -m mini_isp.run --input data/sample.png --out runs --pipeline_mode classic --name v02_m3_chroma \\
  --set stages.denoise.method=chroma_gaussian \\
  --set stages.denoise.sigma_y=1.0 \\
  --set stages.denoise.sigma_c=2.0 \\
  --set stages.denoise.ksize=5

Requirements:
	•	Add repeatable flag: --set KEY=VALUE (may be passed multiple times).
	•	KEY is a dotted path into the resolved config dict (e.g., stages.denoise.method, stages.denoise.sigma_y, metrics.enable).
	•	VALUE must be type-coerced deterministically:
	•	true/false → bool
	•	integers (e.g., 5) → int
	•	floats (e.g., 1.25) → float
	•	null/none → None
	•	otherwise string (preserve as-is)
	•	Overrides are applied after loading config file defaults, so CLI always wins.
	•	Do not change existing behavior when --set is not provided.
	•	Do not change run-folder layout, manifest.json schema, or viewer paths.
	•	Do not edit docs; no new heavy dependencies.

Implementation notes:
	•	Put the parsing + apply logic in a small helper (e.g., mini_isp/config_overrides.py or inside run.py if you prefer), but keep it testable.
	•	For missing intermediate dict keys, create them (e.g., setting stages.denoise.method should work even if stages exists but denoise doesn’t).
	•	If a dotted path traverses a non-dict, exit with a clear error.

Tests (pytest):
	•	Add a unit test that starts from a small base dict and applies multiple --set entries; assert nested keys created and types are correct.
	•	Add an integration-ish test that runs python -m mini_isp.run on data/sample.png with:
	•	--set stages.denoise.method=chroma_gaussian
	•	--set stages.denoise.sigma_y=1.0
	•	--set stages.denoise.sigma_c=2.0
	•	--set stages.denoise.ksize=5
and then asserts the resulting run’s stages/**/debug.json for denoise reports "method": "chroma_gaussian" (or wherever method is recorded in debug).

Deliverables:
	•	Provide one command to run tests: pytest -q
	•	Provide one command to run the pipeline with overrides (no YAML file)

### Patch prompt 2 — debug.json param de-dup (no schema/layout changes)
Task: remove duplicated parameter reporting in debug.json by eliminating debug.metrics.params (and any similar metrics.params occurrences) and keeping all stage configuration under top-level debug.params.
Rules:
	•	do not change run-folder layout, manifest schema, or viewer assets/paths; do not edit docs; minimal changes only.
	•	preserve existing debug.json top-level keys: stage, params, metrics, warnings, notes.
	•	keep metrics strictly numeric/result metrics (min/max/p01/p99, clip flags, n_fixed, gain stats, etc.); no config duplication.
	•	if a stage has “derived/effective” parameters that aren’t part of config, put them under metrics.derived (not metrics.params).
Tests: update/extend any tests that assert debug structure so pytest -q passes.
Deliverables: give (1) the exact command to run tests, and (2) one command to run a pipeline and the file path to inspect to confirm metrics.params is gone.

---

## v0.2-M4 — Sharpen tuning baseline (prove improvement)

### Final prompt
v0.2-M4 — Sharpen tuning baseline (prove improvement)

Implement a sharpen baseline upgrade that measurably improves edge clarity while reducing ringing/halo artifacts, using only NumPy and deterministic logic; keep all existing behavior intact; add a new method sharpen.method: unsharp_mask_tuned (v0.2 default remains unsharp_mask unless clearly justified by tests); do not edit docs; keep run-folder layout + manifest.json schema + viewer paths unchanged; no heavy deps; PNG bootstrap must still work; pytest -q must pass.

Scope (sharpen only)
	•	operates on RGB_LINEAR_F32 → RGB_LINEAR_F32; dtype float32
	•	edge handling: edge-clamp (replicate), consistent with existing convolution helpers
	•	no implicit clipping; if you add optional clipping, it must be explicitly controlled and recorded via clip_applied + clip_range in debug
	•	keep parameters minimal and deterministic; recommended params for unsharp_mask_tuned:
	•	sigma (float; default 1.0)
	•	amount (float; default 0.4)
	•	threshold (float; default 0.01)
	•	luma_only (bool; default true)
	•	gate_mode (string; default “soft”) where “soft” implements smooth thresholding of the mask (no hard cutoff)
	•	implementation guidance for unsharp_mask_tuned:
	•	compute a luma signal Y = 0.2126R + 0.7152G + 0.0722B
	•	build mask from Y: maskY = Y − gaussian_blur(Y)
	•	apply a soft threshold gate to maskY (deterministic, define exactly):
	•	eps = 1e-6
	•	denom = max(threshold, eps)
	•	gain = clip((abs(maskY) − threshold) / denom, 0..1)
	•	gated = maskY * gain
	•	add back to RGB equally: out = rgb + amount * gated[…, None]
	•	if luma_only=false, fall back to existing per-channel behavior but still use the same soft gating
	•	required debug fields (debug.json):
	•	metrics.method = “unsharp_mask_tuned”
	•	metrics.params includes sigma, amount, threshold, luma_only, gate_mode, edge_mode
	•	metrics.clip_applied, metrics.clip_range
	•	keep existing numeric stats (min/max/p01/p99) and any existing diagnostics fields if present
	•	use the existing metrics/diagnostics scaffolding when enabled (v0.2-M2) to show improvement; do not change schemas; only emit additive optional outputs under stage extra/ as already established.

Tests (pytest)
	•	synthetic step-edge test (deterministic):
	•	construct a simple RGB_LINEAR_F32 image with a hard vertical edge (left=0.2, right=0.8), optionally add tiny deterministic noise (fixed seed)
	•	run baseline sharpen.method=unsharp_mask (current default params) and new sharpen.method=unsharp_mask_tuned (defaults)
	•	assert shape/dtype unchanged; values finite
	•	define two deterministic metrics measured in a narrow band around the edge:
	•	use band width = 5 pixels on either side of the edge (10px total window)
	•	edge_strength: mean gradient magnitude around the edge (should be >= baseline * (1 - tol), set tol explicitly, e.g., 0.05)
	•	overshoot: max(out_band) − max(in_band) and min(in_band) − min(out_band) (tuned should be <= baseline)
	•	pass condition: tuned method reduces overshoot vs baseline while preserving edge_strength within tolerance (make thresholds explicit in the test).

Deliverables
	•	end with (1) one command to run pytest -q; and (2) two commands to run the pipeline for A/B compare runs using only CLI –set overrides:
	•	baseline sharpen: –set stages.sharpen.method=unsharp_mask (and any baseline params you want fixed)
	•	tuned sharpen: –set stages.sharpen.method=unsharp_mask_tuned (and tuned params)
	•	use CLI –set overrides for A/B runs; do not require creating new YAML config files.

---

## v0.2-M5 — Optional real RAW/DNG input support (validation)

### Final prompt
v0.2-M5 — Optional real RAW/DNG input support (validation)

Add optional RAW/DNG input loading so the pipeline can run on a true Bayer mosaic, while keeping existing PNG bootstrap behavior unchanged. RAW support must be optional (extra dependency only when used).

Requirements

Input modes:
	•	if input is .png (or default), keep current PNG-bootstrap path unchanged;
	•	if input is RAW/DNG (e.g., .dng, .nef, .cr2, .arw), load as a single-channel Bayer mosaic (2D H×W);

RAW loader (rawpy):
	•	implement a new loader path that uses rawpy (lazy import);
	•	extract the mosaic as a 2D array (e.g., raw.raw_image_visible or raw.raw_image depending on availability);
	•	derive bit_depth as follows (deterministic priority order):
	1.	if rawpy exposes a reliable max value / white level (e.g., raw.white_level), compute bit_depth = ceil(log2(white_level + 1));
	2.	else fallback to dtype bits (uint16→16, etc.);
	•	derive meta.cfa_pattern deterministically from rawpy when available:
	•	if raw.raw_pattern (2×2 ints) and raw.color_desc are present, map as:
	•	desc = raw.color_desc decoded to string (e.g., “RGBG”);
	•	pattern = 2×2 array of ints where each int indexes into desc;
	•	convert each entry to a channel letter via desc[idx] (e.g., 0→”R”, 1→”G”, 2→”B”);
	•	read out in row-major order to a 4-letter string (top-left, top-right, bottom-left, bottom-right), e.g., “RGGB”;
	•	set meta.cfa_pattern to that string;
	•	else fallback to config input.bayer_pattern (default “RGGB”);

Metadata propagation:
	•	populate meta.cfa_pattern, meta.black_level, meta.white_level, meta.bit_depth (when known);
	•	ensure manifest.input records cfa_pattern and (when available) bit_depth, black_level, white_level for RAW inputs;
	•	for PNG inputs, keep current behavior unchanged (these fields may be absent or null); do not break existing runs/viewer;

Normalization (must be explicit and deterministic):
	•	normalize to RAW_BAYER_F32 via:
norm = (raw - black_level) / max(white_level - black_level, eps)
where eps = 1e-6;
	•	then clip to [0,1];
	•	output dtype float32, shape H×W preserved;

Dependencies:
	•	use rawpy only when RAW is used (lazy import / optional install);
	•	do not add rawpy to default runtime requirements; add requirements-raw.txt;

Compatibility:
	•	run-folder layout, manifest.json schema, viewer paths unchanged;
	•	pytest -q must pass;
	•	do not edit docs;

Tests (pytest):
	•	unit test (no rawpy): verify normalization logic on a synthetic Bayer array (dtype float32, 2D shape, range clipped [0,1], CFA fallback works);
	•	rawpy integration test:
	•	if rawpy not installed: skip;
	•	if installed: mock rawpy.imread (or the context) to return deterministic mosaic + metadata (black_level/white_level/raw_pattern/color_desc); assert:
	•	cfa_pattern is derived from raw_pattern+color_desc correctly,
	•	normalized mosaic matches expected math + clipping,
	•	manifest.input fields are populated as specified;

Deliverables:
	•	one command to install RAW optional dependency (pip install -r requirements-raw.txt);
	•	one command to run pytest -q;
	•	one command to run pipeline on a RAW/DNG input; and one command to run pipeline on PNG to show unchanged behavior.

### Patch prompt 1 — BGGR/GRBG/GBRG CFA support in wb_gains (+ demosaic if needed)
v0.2-M5 Patch — support BGGR/GRBG/GBRG CFA in wb_gains (+ demosaic if needed)

Task:
	•	Extend wb_gains to support CFA patterns RGGB, BGGR, GRBG, GBRG when operating on RAW_BAYER_F32.
	•	If demosaic currently assumes RGGB, extend it to support the same four CFA patterns (for both bilinear and malvar if malvar exists).

Rules / Constraints:
	•	Do not change run-folder layout, manifest.json schema, viewer assets/paths.
	•	Do not edit docs.
	•	Keep dependencies minimal (NumPy only).
	•	Keep PNG bootstrap behavior unchanged; CFA can remain default RGGB for PNG input.
	•	pytest -q must pass.

Implementation notes:
	•	Implement a single CFA mapping helper (e.g., cfa_index_map(pattern)) that returns the (row_parity, col_parity) positions for R/G/B sites.
	•	In wb_gains, apply gains by indexing into the mosaic using that mapping; apply G gain to both green sites.
	•	In demosaic, use the same mapping to place known samples into the right channels before interpolation; keep dtype float32.
	•	Use a fixed 2×2 parity map at the top-left: RGGB: R(0,0), Gr(0,1), Gb(1,0), B(1,1); BGGR: B(0,0), Gb(0,1), Gr(1,0), R(1,1); GRBG: Gr(0,0), R(0,1), B(1,0), Gb(1,1); GBRG: Gb(0,0), B(0,1), R(1,0), Gr(1,1); wb_gains applies G gain to both green sites.

Tests:
	•	Add unit tests for each CFA pattern:
	1.	Build a tiny synthetic 4×4 mosaic where R/G/B sites have distinct constants; apply wb gains (e.g., r=2, g=3, b=5) and assert only the correct sites were scaled.
	2.	Demosaic sanity: output shape H×W×3 float32; and a simple check that swapping pattern changes which channel receives which site (i.e., pattern is actually honored).
	•	Add an integration-ish test: run the pipeline on the existing sample.dng (BGGR) if present, or mock the RAW loader to return meta.cfa_pattern=“BGGR”, and assert the pipeline completes past wb_gains.

Deliverables:
	•	One command to run tests (pytest -q).
	•	One command to run: python -m mini_isp.run --input data/sample.dng --out runs --pipeline_mode classic --name raw_demo (or with whatever RAW file you have).

### Patch prompt 2 — RAW white-balance support (rawpy)
v0.2-M5 Patch — RAW white-balance support (rawpy)

Context:
	•	Previous RAW runs used unity gains [1,1,1], which can produce a green cast. This patch reads camera/daylight WB from rawpy in the RAW/DNG path and feeds effective gains into wb_gains by default (PNG bootstrap unchanged).

Task:
	•	In the RAW/DNG input path, obtain WB gains from rawpy and pass them into wb_gains when no explicit wb_gains are set.

Priority order (deterministic):
	1.	Use raw.camera_whitebalance (as-shot) when available.
	2.	Else use raw.daylight_whitebalance (preset daylight WB).
	3.	Else fall back to unity [1.0, 1.0, 1.0].

Behavior:
	•	Normalize gains so green averages to 1.0:
	•	use the average of the two green positions in the CFA as the normalization base
	•	scale R and B accordingly
	•	Apply these normalized gains in wb_gains for RAW inputs (unless overridden by config).
	•	Record the chosen source and effective gains in:
	•	wb_gains stage debug.json:
	•	params.wb_gains
	•	params.wb_source = "camera_whitebalance" | "daylight_whitebalance" | "unity"
	•	meta.wb_gains and meta.wb_applied (if those fields already exist)
	•	any existing input metadata in the manifest that tracks WB (if present)

Constraints:
	•	No changes to run-folder layout, manifest.json schema, viewer assets/paths, or docs.
	•	rawpy remains optional (lazy import); no new heavy dependencies.
	•	PNG bootstrap behavior unchanged.
	•	pytest -q must pass.

Tests (pytest):
	•	Use mocked rawpy (no real RAW needed).
	1.	camera_whitebalance present:
	•	gains normalized (green ≈ 1.0)
	•	wb_source == "camera_whitebalance"
	2.	Only daylight_whitebalance present:
	•	normalized gains from daylight WB
	•	wb_source == "daylight_whitebalance"
	3.	Neither present:
	•	gains [1.0, 1.0, 1.0]
	•	wb_source == "unity"
	•	Where practical, assert:
	•	meta.wb_gains matches the effective gains
	•	wb_gains debug.json.params.wb_gains matches the effective gains

Deliverables:
	•	One command to run tests (pytest -q).
	•	One command to run: python -m mini_isp.run --input data/sample.dng --out runs --pipeline_mode classic --name raw_demo

### Patch prompt 3 — CLI WB toggles for wb_gains (meta/unity/manual)
v0.2-M5 Patch — CLI WB toggles for wb_gains (meta/unity/manual)

Context:
	•	For RAW tuning and debugging/validation, it’s useful to force WB “off” (unity gains) or set manual gains without writing a YAML config.
	•	The existing CLI `--set` override mechanism is scalar-only, which makes passing a 3-value gains list awkward.

Task:
	•	Add CLI flags to `python -m mini_isp.run`:
	•	`--wb-mode {meta,unity,manual}`
	•	`--wb-gains R G B` (space-separated triple; required when `--wb-mode manual`)

Precedence / defaults:
	•	Preserve existing `--config` behavior; do not change behavior when these flags are not provided.
	•	Explicit stage config wins (YAML or `--set`):
	•	If `stages.wb_gains.wb_gains` (or `stages.wb_gains.gains`) is present in the resolved config, ignore `--wb-mode/--wb-gains`.
	•	Default wb-mode:
	•	RAW/DNG inputs → `meta`
	•	PNG bootstrap inputs → `unity`

Behavior:
	•	`meta`: use WB gains derived from RAW metadata (rawpy) exactly as in v0.2-M5 Patch prompt 2.
	•	`unity`: force gains `[1.0, 1.0, 1.0]` regardless of RAW metadata.
	•	`manual`: parse `--wb-gains R G B` as floats and apply as `[r, g, b]` (apply G to both green sites).
	•	Record in `wb_gains` stage output (no schema/layout changes):
	•	`debug.json.params.wb_mode`
	•	`debug.json.params.wb_source` (actual provenance of applied gains, not the intent):
	•	when `wb_mode=meta`: `camera_whitebalance|daylight_whitebalance|unity_fallback`
	•	when `wb_mode=unity`: `unity`
	•	when `wb_mode=manual`: `manual`
	•	`debug.json.params.wb_gains` (effective applied gains)
	•	Propagate to `frame.meta` consistently (e.g., `meta.wb_gains`, `meta.wb_applied`, and `meta.wb_source` if present).

Constraints:
	•	No changes to run-folder layout, manifest.json schema, viewer assets/paths, or docs.
	•	No new heavy dependencies.
	•	pytest -q must pass.

Tests (pytest):
	•	Unit test: starting from a small config dict, ensure `--wb-mode/--wb-gains` are applied only when no explicit `stages.wb_gains.wb_gains` is set.
	•	Integration-ish test: run `python -m mini_isp.run` on PNG bootstrap with `--wb-mode unity` and assert `wb_gains` debug reports `wb_mode="unity"` and gains `[1,1,1]`.
	•	Unit test: `--wb-mode manual` with no `--wb-gains` errors with a clear message.

Deliverables:
	•	One command to run tests: pytest -q
	•	Example:
python -m mini_isp.run --input data/sample.dng --out runs --pipeline_mode classic --name raw_demo_unity --wb-mode unity

---

## v0.2-M6 — RAW crop utility (testing support)

### Final prompt
v0.2-M6 — RAW crop utility (testing support)

Add a small, optional CLI utility to generate deterministic Bayer RAW crops for fast ISP iteration and reproducible A/B comparisons; this is a developer/testing tool only (not a pipeline stage); implement it under mini_isp/tools/ and expose it as a module entrypoint.

Task
	•	create a new package folder mini_isp/tools/ (with init.py) if it doesn’t exist; add raw_crop.py inside it
	•	add a CLI entrypoint so users can run: python -m mini_isp.tools.raw_crop …

Requirements
	•	input: real RAW/DNG using the same RAW loader path as the pipeline (rawpy; lazy import); reuse existing loader utilities where possible (do not duplicate metadata logic if it already exists)
	•	output: a cropped Bayer mosaic (2D H×W) plus a minimal metadata sidecar; this tool must not run demosaic/WB/tone or any pipeline stages
	•	determinism: crop is purely index-based with explicit –x –y –w –h (no auto ROI selection); output must be deterministic for a given input + coordinates
	•	scope: this is not a pipeline stage; do not change run-folder layout, manifest.json, or viewer assets/paths; do not edit docs
	•	dependencies: rawpy is an optional dependency used only when RAW/DNG input is provided (lazy import); do not add rawpy to default requirements.txt (keep optional raw deps separate if the repo already follows that convention)
	•	safety/validation:
	•	validate that crop bounds are within the mosaic shape; error clearly if out-of-bounds
	•	create OUT_DIR if missing; default overwrite=false unless –overwrite is passed
	•	preserve CFA correctness: if cropping starts at odd x/y, the effective CFA alignment changes; handle this explicitly by adjusting the reported cfa_pattern in meta.json so that downstream stages interpret the crop correctly

CLI
Command: python -m mini_isp.tools.raw_crop
Flags (required unless noted):
	•	–input PATH
	•	–out OUT_DIR
	•	–x X0
	•	–y Y0
	•	–w W
	•	–h H
	•	–dtype float32|uint16 (default: float32)
	•	–overwrite (optional; default false)

Outputs (exact filenames in OUT_DIR)
	•	crop.npy  (mosaic; 2D; dtype per –dtype)
	•	meta.json (sidecar metadata)

Metadata sidecar schema (meta.json; must include these keys)
	•	source_path, x, y, w, h
	•	mosaic_shape (full mosaic H×W), crop_shape (h×w)
	•	cfa_pattern (after any parity adjustment from x/y)
	•	black_level, white_level, bit_depth

Notes on dtype
	•	dtype=float32: store normalized RAW in [0,1] using the same normalization formula as the pipeline raw loader (with safe divide + clip); record in meta.json that the crop is normalized float32
	•	dtype=uint16: store the original mosaic values as uint16 without normalization; record in meta.json that the crop is unnormalized uint16 and include black/white/bit_depth for later normalization

Tests (pytest)
	•	unit test (no rawpy): cropping on a synthetic mosaic array to verify:
	•	output shape matches –w/–h
	•	determinism (same input + coords → same output)
	•	out-of-bounds raises a clear error
	•	CFA parity adjustment works: changing x/y parity updates cfa_pattern as expected
	•	integration test (rawpy optional):
	•	if rawpy not installed: skip
	•	if installed: mock rawpy.imread (or the project’s raw loader entrypoint) to return a deterministic mosaic + metadata; verify:
	•	crop.npy content and dtype
	•	meta.json keys and values (including cfa_pattern, black/white/bit_depth)
	•	pytest -q must pass

Constraints
	•	do not change run-folder layout, manifest.json schema, or viewer assets/paths
	•	do not add new heavy dependencies
	•	do not edit docs
	•	keep behavior deterministic

Deliverables
	•	end with:
(1) one command to install optional RAW deps (if needed; e.g., pip install -r requirements-raw.txt or pip install rawpy)
(2) one command to run pytest -q
(3) one example CLI command that generates a crop from a RAW/DNG file

### Patch prompt 1 — Add crop.npy + meta.json input path
v0.2-M6 Patch — Add crop.npy + meta.json input path
Add a dev-only input option to python -m mini_isp.run so --input may point to a Bayer mosaic saved by the RAW crop utility (crop.npy + meta.json). This is a third input type alongside PNG bootstrap and RAW/DNG.

Task
	•	Extend input handling in mini_isp.run: if --input ends with .npy, load it as a RAW Bayer mosaic; require meta.json in the same directory (Path(input).with_name("meta.json")).
	•	Use meta.json to populate cfa_pattern, black_level, white_level, bit_depth (and validate crop info keys x,y,w,h exist).
	•	Treat this path like the RAW mosaic path after ingestion: preserve pipeline behavior downstream of raw_norm; do not change run-folder layout, manifest schema, viewer, or docs.

Input contract
	•	crop.npy: 2D array H×W; dtype either float32 (already normalized ~[0,1]) or uint16 (sensor code values).
	•	meta.json required keys: cfa_pattern, black_level, white_level, bit_depth, x, y, w, h. (Optional: source_path for provenance.)

Behavior
	•	If crop.npy is float32: assume normalized; clip to [0,1].
	•	If crop.npy is uint16: normalize via (raw - black) / max(white - black, eps) with eps=1e-6; clip to [0,1].
	•	Propagate metadata the same way the RAW/DNG path does: populate frame meta and the existing manifest input fields for CFA/levels/bit depth.

Validation
	•	Error if meta.json missing or required keys absent; error if array is not 2D; error if dtype is not float32 or uint16.

Tests (pytest)
	•	Unit: load synthetic .npy + meta.json for both float32 and uint16; assert correct dtype/shape, normalization+clipping for uint16, and metadata propagation matches expectations.
	•	Integration: run mini_isp.run with a temporary crop.npy + meta.json; assert the run succeeds and expected stage artifacts exist (at minimum 00_raw_norm/debug.json).

Constraints
	•	No changes to run-folder layout, manifest.json schema, viewer assets/paths, or docs; no new heavy deps; pytest -q must pass.

Deliverables
	•	One command to run tests: pytest -q
	•	One example command:
python -m mini_isp.run --input crops/raw_demo/crop.npy --out runs --pipeline_mode classic --name crop_demo

---

## v0.2-M7 — Scene-pack runner + consolidated JSON report

### Final prompt
v0.2-M7 — Scene-pack runner + consolidated JSON report

Add a small CLI tool to run a folder of inputs under two configurations (baseline vs candidate) and write a single JSON report summarizing metrics per scene. Developer/testing only; no changes to pipeline stages, run layout, manifest.json, or viewer.

CLI
	•	Implement python -m mini_isp.tools.scene_pack with:
	•	--inputs DIR (required): folder of inputs (.png, RAW/DNG, crop.npy).
	•	--out DIR (required): base output dir (e.g. runs/scene_pack_demo).
	•	--name NAME (optional): report name stem, default scene_pack.
	•	Baseline config (required): either --baseline-config PATH or one/more --baseline-set KEY=VALUE (same semantics as existing --set).
	•	Candidate config (required): either --candidate-config PATH or one/more --candidate-set KEY=VALUE.
	•	--metrics (default true): ensure metrics/diagnostics are enabled for both runs.
	•	--ext (optional): comma-separated extension filter (default supports .png,.dng,.nef,.cr2,.arw,.npy).
	•	--skip-errors (optional): skip inputs that fail validation (e.g., .npy missing meta.json).

For each matching file:
	•	Run baseline and candidate via the existing pipeline runner, with deterministic run names like <name>/<file_stem>__baseline and __candidate.
	•	Use existing loaders (PNG, RAW/DNG via rawpy, crop.npy + meta.json). For .npy inputs, meta.json must exist in the same directory.

Report format
	•	Write ${out}/scene_pack_${name}.json with:
	•	top-level: scene_pack, created_utc, inputs (list).
	•	each inputs[i] contains:
	•	input_path
	•	baseline and candidate blocks with: label, run_dir, pipeline_mode, runtime_ms, metrics (small numeric dict).
	•	diff: for each metric present in both, <metric>_delta = candidate - baseline.
	•	Implement a small helper (e.g. extract_report_metrics(...)) that chooses which numeric fields from existing metrics JSONs to include (start with a few like PSNR/L1 diff; easy to extend later).
	•	Keep output deterministic (inputs sorted by filename; report stable for same inputs/configs).

Constraints
	•	Do not change run-folder layout, manifest.json schema, or viewer behavior.
	•	Reuse existing metrics/diagnostics artifacts; no new heavy deps (NumPy + stdlib only).
	•	pytest -q must pass.

Tests (pytest)
	•	Use a temp dir with 2–3 tiny PNGs; run scene_pack with baseline/candidate via --*-set.
	•	Assert the JSON report exists and has the expected structure:
	•	scene_pack, created_utc, inputs list
	•	for each entry: input_path, baseline.run_dir, candidate.run_dir, runtime_ms, metrics dicts, and diff dict.
	•	run folders actually exist.
	•	Determinism: run scene_pack twice with the same args and assert the parsed JSONs are identical (you may ignore created_utc if you want).

Deliverables
	•	one command to run tests (pytest -q)
	•	one example command, e.g.:
python -m mini_isp.tools.scene_pack \
  --inputs data/scenes \
  --out runs/scene_pack_demo \
  --name denoise_tweak_01 \
  --baseline-set stages.denoise.method=gaussian \
  --candidate-set stages.denoise.method=chroma_gaussian

---

## v0.2-M8 — DNG-aware CCM auto-default (resolver + deterministic DNG source)
Implementation note: Patch 1 and Patch 2 were implemented together in one commit.

### Patch prompt 1 — CCM resolver policy scaffold (no DNG math port yet)
Add runner-side CCM auto-default policy wiring, without changing stage list or adding new pipeline modes.

Scope
	•	Keep stage name `ccm` and stage order unchanged.
	•	No run-layout / manifest / viewer schema changes.
	•	Treat this as resolver policy (runner), not a new stage mode.

Policy / precedence
	•	Define explicit CCM config as presence of any key under resolved `stages.ccm`:
	•	`mode`, `matrix`, `cam_to_xyz_matrix`, `xyz_to_working_matrix`
	•	If explicit, never auto-override.
	•	When not explicit:
	•	For DNG input only, if metadata provides both `cam_to_xyz_matrix` and `xyz_to_working_matrix`, auto-set `ccm.mode=chain` with those matrices.
	•	For non-DNG RAW, do not auto-inject CCM from metadata in this patch.
	•	Otherwise keep identity behavior (no matrix guessing).

Debug / provenance
	•	In `stages/<nn>_ccm/debug.json` under `params`, include:
	•	`auto_default_applied: true|false`
	•	`auto_default_reason` (e.g., `explicit_stage_config|applied_from_dng_tags|missing_dng_ccm|non_dng_raw|non_raw_input`)
	•	`cam_to_xyz_source` and `xyz_to_working_source` when available
	•	Keep existing chain debug fields and `meta.ccm` / `meta.ccm_mode`.

Tests
	•	Resolver unit tests:
	•	explicit keys win over auto-default
	•	DNG + available matrices auto-selects chain
	•	non-DNG RAW does not auto-select chain
	•	missing metadata matrices falls back to identity behavior with reason

### Patch prompt 2 — Deterministic DNG tags + D50 adaptation source
Add DNG metadata extraction that feeds Patch-1 resolver with deterministic matrices.

Scope
	•	DNG only (`.dng`) in this patch.
	•	No `_refcheck` references in mini-ISP docs/code.
	•	No stage split; continue using `ccm.mode=chain`.

DNG metadata matrix source
	•	Read DNG tags (via exif metadata extraction) for:
	•	`ForwardMatrix1/2`
	•	`ColorMatrix1/2`
	•	`CameraCalibration1/2` (optional)
	•	`AnalogBalance` (optional)
	•	`AsShotNeutral` + `CalibrationIlluminant1/2` (for interpolation weight)
	•	Selection order:
	•	Prefer interpolated `ForwardMatrix` when available.
	•	Else synthesize deterministic fallback from native chain (`AnalogBalance * CameraCalibration * ColorMatrix`) and derive forward matrix per DNG-spec default-FM rule.
	•	Reject non-finite / near-zero matrices; on failure return unavailable with reason.

D50 adaptation / chain wiring
	•	Treat DNG-derived `cam_to_xyz_matrix` as camera→XYZ(D50).
	•	Set `xyz_to_working_matrix` to a built-in constant XYZ(D50)→linear sRGB(D65) (Bradford-adapted).
	•	Resolver then applies `mode=chain` with:
	•	`effective_matrix = xyz_to_working_matrix @ cam_to_xyz_matrix`
	•	`out = in @ effective_matrix.T`

Fallback policy
	•	If DNG tag derivation is unavailable/invalid: do not auto-enable chain; remain identity with recorded reason.
	•	Do not auto-fallback to `rawpy.rgb_xyz_matrix` in M8.

Tests
	•	Metadata-only unit tests for DNG matrix derivation:
	•	ForwardMatrix interpolation path
	•	ColorMatrix native-chain synthesized fallback path
	•	invalid metadata handling (unavailable + reason)
	•	Integration-ish resolver test that consumes DNG metadata fields and produces chain config.

Deliverables
	•	`pytest -q`
	•	Example:
python -m mini_isp.run --input data/sample.dng --out runs --pipeline_mode classic --name v02_m8_dng_auto

---

## v0.3-M1 — Diagnostics surfaced in viewer (non-breaking)

### Final prompt
v0.3-M1 — Diagnostics surfaced in viewer (non-breaking)

Expose existing metrics/diagnostics in the static viewer UI (single-run and compare modes), without changing run layout, manifest.json, or diagnostics file formats.

Scope
	•	Viewer only: static index.html / app.js / styles.css (no build tooling).
	•	Use existing artifacts under each stage folder, typically:
	•	stages/<nn>_<name>/extra/metrics.json
	•	stages/<nn>_<name>/extra/diff_metrics.json (if present)
	•	stages/<nn>_<name>/extra/diagnostics/false_color.png, zipper.png, halo.png (diagnostic images, if present)

Behavior
	•	Metrics panel
	•	Add a small “Metrics” panel in the right-hand column (below or near the existing debug JSON).
	•	For the current stage:
	•	In single-run mode: load metrics.json (if it exists) and render a simple key→value table for the top-level numeric fields (e.g., min, max, p01, p99, luma_mean, clip_pct, etc.).
	•	In compare mode: show A and B side-by-side in the same panel (two columns, same keys where possible).
	•	If metrics.json is missing: show a small “Metrics: N/A” message instead of throwing.
	•	Diff metrics section
	•	If diff_metrics.json exists for the current stage, show a small “Diff metrics” subsection.
	•	Display key fields such as l1, l2, psnr (and any other top-level numeric fields) in a simple table.
	•	In compare mode, only show diff metrics that actually exist (no extra logic needed beyond null-checking).
	•	Diagnostics viewer
	•	Add a “Diagnostics” area that lists available diagnostic images for the current stage:
	•	Look for false_color.png, zipper.png, halo.png under extra/diagnostics/ (fallback to extra/ if needed).
	•	For single-run mode: allow the user to toggle between the main preview and each diagnostic image (e.g., via small buttons or a dropdown).
	•	For compare mode: when a diagnostic is selected, show A and B diagnostics side-by-side if both exist; if only one side has the image, show that side and mark the other as “N/A”.
	•	If no diagnostics are present for the stage, show a simple “Diagnostics: N/A” label (no errors).
	•	Graceful behavior
	•	All new UI must fail gracefully when some or all artifacts are missing:
	•	no exceptions in JS if a JSON or PNG is absent
	•	clear “N/A” labels instead of blank/broken UI
	•	Existing viewer interactions (stage selection, preview, debug display, compare sync) must continue to work as before.

Constraints
	•	Do not change:
	•	run-folder layout
	•	manifest.json schema
	•	metrics/diagnostics file formats or paths
	•	No new dependencies or build tools; keep everything as plain JS/HTML/CSS.
	•	Keep code deterministic and simple (no async race conditions between A/B).

Tests
	•	If you already have a JS test harness, add minimal tests for:
	•	metric panel rendering with a small fake metrics.json
	•	diagnostics toggle logic with a fake diagnostics set
	•	If not, you may skip JS tests for now; Python tests remain unchanged.
	•	pytest -q must still pass.

Deliverables
	•	One command to run tests:
pytest -q
	•	One viewer URL example for single-run (e.g. http://localhost:8000/viewer/index.html?manifest=...).
	•	One viewer URL example for compare mode (e.g. ...?compare=compare_bundle.json).

## v0.3-M2 — Classical tone/DRC refinement (parametric Reinhard)

### Final prompt
v0.3-M2 — Classical tone/DRC refinement (parametric Reinhard)

Turn tone.method=reinhard into a tunable classical tone-mapping curve and validate improvements using compare mode, metrics/diagnostics, the v0.3-M1 viewer panels, and the scene-pack runner. Focus on better highlight handling and local contrast without introducing new artifacts. No schema/layout changes.

Requirements
	•	Stage: tone only (RGB_LINEAR_F32 → RGB_LINEAR_F32); keep outputs deterministic; no new deps; no changes to run layout, manifest.json schema, or viewer asset paths; no docs edits.
	•	Parametrize Reinhard under stages.tone with a small set of scalar params:
	•	exposure (float, default 1.0)
	•	white_point (float, default 1.0, chosen so defaults match the previous x/(1+x) behavior)
	•	gamma (float, default 1.0)
	•	Base Reinhard curve: operate in linear RGB per channel with
	x_scaled = exposure * image;
	y = x_scaled / (1.0 + x_scaled / white_point);
	out = y ** (1.0 / gamma) (all float32, no implicit clipping).
	•	Keep defaults stable and record all params cleanly under debug.params.
	•	If clipping is used, make it opt-in and record clip_applied + clip_range in debug.
	•	Viewer metrics panel: keep consuming the full metrics JSON, but:
	•	show a small default subset (e.g. luma_mean, clip_pct, p99, PSNR when available),
	•	add a “Show all metrics” toggle in the metrics panel header that expands to the full metrics set for both sides in compare mode,
	•	default to the subset view on each page load (no persistence across sessions).

Tests (pytest)
	•	Synthetic test on a small HDR-like ramp: verify the parametric Reinhard curve is monotonic, compresses highlights (bright inputs reduced more than mids), preserves ordering (no inversions), and keeps dtype/shape unchanged with finite values.

Deliverables
	•	python -m pytest -q
	•	One scene-pack run using mini_isp.tools.scene_pack + --set overrides comparing baseline vs tuned tone.method=reinhard across a small set of scenes; tuned config should show consistent metric + visual improvements with no regressions in dark regions or skin tones.
	•	(Optional, but recommended): two mini_isp.run calls on a representative single scene (baseline vs tuned Reinhard) and the corresponding viewer URLs, to manually inspect tone behavior with the updated metrics panel.
