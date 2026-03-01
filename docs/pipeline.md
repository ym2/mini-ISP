# Pipeline

mini-ISP is organized as a **sequence of stages** with explicit contracts. A stage consumes a `Frame` (image + metadata) and returns a new `Frame`, plus optional debug artifacts.

This doc defines:
- the **current stage set**,
- the default **stage order** per `pipeline_mode`,
- and where optional hooks (skin mask) fit.

## Canonical stage names

### Separate stages
- `raw_norm` — RAW ingest + normalization (black level / white level), optional clamp; expects Bayer-mosaic RAW input
- `dpc` — defect pixel correction (RAW-domain)
- `lsc` — lens shading correction (RAW-domain gain)
- `wb_gains` — apply white-balance gains (still RAW-domain)
- `demosaic` — Bayer → RGB (bilinear; optional malvar stub)
- `denoise` — RGB denoise baseline (gaussian/box; chroma_gaussian available), upgrade path to more advanced methods
- `ccm` — 3×3 color correction matrix in linear RGB (identity, provided, or resolver-injected chain config)
- `stats_3a` — compute AE/AWB/AF-style stats (**non-invasive**, does not change pixels)
- `tone` — tone mapping / dynamic range compression (Reinhard/filmic now; **AI-DRC** can be added later as another method)
- `color_adjust` — post-tone color adjustment (sat/hue/roll-off or LUT-like; identity by default)
- `sharpen` — unsharp mask (baseline + tuned variant)
- `oetf_encode` — output encode (default: sRGB 8-bit), extendable to other encodes / bit depths later

### Curated composites
- `jdd_raw2rgb` — joint demosaic + denoise composite (RAW mosaic → linear RGB)
- `drc_plus_color` — curated tone/DRC + color adjustment composite

### Optional hook (side signal)
- `skin_mask` — optional mask generation (heuristic now; seg-model later). Not necessarily in the main stage list; called by stages that opt in.

## pipeline_mode definitions

### `classic` (separate stages)
Recommended for learning and debugging.

1. `raw_norm`
2. `dpc`
3. `lsc`
4. `wb_gains`
5. `demosaic`
6. `denoise`
7. `ccm`
8. `stats_3a`
9. `tone`
10. `color_adjust`
11. `sharpen`
12. `oetf_encode`

### `jdd` (joint demosaic + denoise)
Curated composite: replaces `demosaic` + `denoise`.

1. `raw_norm`
2. `dpc`
3. `lsc`
4. `wb_gains`
5. `jdd_raw2rgb`
6. `ccm`
7. `stats_3a`
8. `tone`
9. `color_adjust`
10. `sharpen`
11. `oetf_encode`

### `drc_plus` (tone/DRC + color adjustment)
Curated composite: replaces `tone` + `color_adjust` with `drc_plus_color`.

1. `raw_norm`
2. `dpc`
3. `lsc`
4. `wb_gains`
5. `demosaic`
6. `denoise`
7. `ccm`
8. `stats_3a`
9. `drc_plus_color`
10. `sharpen`
11. `oetf_encode`

## Where the skin-mask hook fits

The skin mask is an **optional side signal** used to modulate parameters (no major branching in the current pipeline).

Recommended behavior:
- Compute `skin_mask` once after RGB exists (e.g., after `ccm`).
- Use it initially in **one stage only** (recommended: `color_adjust` or `drc_plus_color`).
- Default config keeps it **off**, and even when enabled uses gentle deltas.

## RAW input constraint
- The RAW pipeline path targets Bayer mosaics (2D sensor mosaic data).
- RGB/non-mosaic DNG payloads are not supported in this path and are rejected at input load.

## CCM resolver policy (current)
- Stage list/order is unchanged; CCM policy is resolved in runner config before the `ccm` stage executes.
- Explicit `stages.ccm.*` keys win over auto-default policy.
- RAW auto-default hierarchy:
  1) DNG RAW: runner may auto-inject `ccm.mode=chain` from deterministic DNG metadata-derived matrices.
  2) non-DNG RAW: runner may auto-inject `ccm.mode=chain` from metadata policy `non_dng_meta_default`.
- Non-DNG matrix-source selection (v0.2-M12):
  - source policy `wp_error_min_det` selects `non_dng_cam_to_xyz_matrix` from `rawpy.rgb_xyz_matrix` candidates via metadata-only whitepoint error minimization.
  - selected matrix/source are then consumed by the non-DNG resolver branches below.
- Current non-DNG deterministic rule (no reference scoring): `wp_infer_clean_d65_d50_else_daylight_with_outlier_identity`.
  - clean D65 (`wp_err_d65 < 0.05`) -> `selected_input|d65`
  - clean D50 (`wp_err_d50 < 0.04`) -> `selected_input|d50adapt`
  - ambiguous (`min_err <= 0.08`) with daylight WB -> `pre_unwb_daylight|d65`
  - high-error outlier confidence trigger (`min(wp_err_d50, wp_err_d65) > 0.33`) -> identity fallback (skip auto chain)
  - otherwise -> `pre_unwb_daylight|d65` (or `selected_input|d50adapt` when daylight WB is unavailable)
- If metadata matrices are unavailable/invalid, runner falls back to identity behavior with recorded reason in stage debug params.
- Debug provenance for non-DNG auto-default includes `ccm_source`, `non_dng_meta_rule`, `non_dng_meta_input_variant`, `non_dng_meta_wp_variant`, `non_dng_meta_branch`, `non_dng_meta_selection_reason`, `non_dng_meta_wp_err_d50`, `non_dng_meta_wp_err_d65`, `non_dng_meta_outlier_confidence_threshold`, `non_dng_meta_outlier_confidence_trigger`, `non_dng_meta_outlier_fallback_applied`, and matrix-source details (`non_dng_matrix_source_policy`, `non_dng_matrix_selected_source_variant`, `non_dng_matrix_selected_wp_err_*`, `non_dng_matrix_candidate_count`).

## What’s missing vs a “full ISP”
mini-ISP includes a realistic single-frame skeleton, but does not yet include:
- MFNR / HDR merge (multi-frame pipeline containers)
- advanced artifact controls (false color / moiré suppression, de-ring)
- explicit gamut mapping / richer look LUT stages
- closed-loop AE/AWB/AF control (stats exist; control loop is out of scope for now)
