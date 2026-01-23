# Pipeline

mini-ISP is organized as a **sequence of stages** with explicit contracts. A stage consumes a `Frame` (image + metadata) and returns a new `Frame`, plus optional debug artifacts.

This doc defines:
- the **v0.1 stage set**,
- the default **stage order** per `pipeline_mode`,
- and where optional hooks (skin mask) fit.

## Canonical stage names (v0.1)

### Separate stages
- `raw_norm` — RAW ingest + normalization (black level / white level), optional clamp
- `dpc` — defect pixel correction (RAW-domain)
- `lsc` — lens shading correction (RAW-domain gain)
- `wb_gains` — apply white-balance gains (still RAW-domain)
- `demosaic` — Bayer → RGB (e.g., Malvar)
- `denoise` — RGB denoise baseline (NLMeans), upgrade path to BM3D / edge-aware denoise
- `ccm` — 3×3 color correction matrix in linear RGB (identity or provided)
- `stats_3a` — compute AE/AWB/AF-style stats (**non-invasive**, does not change pixels)
- `tone` — tone mapping / dynamic range compression (Reinhard/filmic now; **AI-DRC** can be added later as another method)
- `color_adjust` — post-tone color adjustment (sat/hue/roll-off or LUT-like; identity by default)
- `sharpen` — unsharp mask with basic edge/halo controls
- `oetf_encode` — output encode (default: sRGB 8-bit), extendable to other encodes / bit depths later

### Curated composites
- `jdd_raw2rgb` — joint demosaic + denoise composite (RAW mosaic → linear RGB)
- `drc_plus_color` — curated tone/DRC + color adjustment composite

### Optional hook (side signal)
- `skin_mask` — optional mask generation (heuristic now; seg-model later). Not necessarily in the main stage list; called by stages that opt in.

## pipeline_mode definitions (v0.1)

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

The skin mask is an **optional side signal** used to modulate parameters (no major branching in v0.1).

Recommended v0.1 behavior:
- Compute `skin_mask` once after RGB exists (e.g., after `ccm`).
- Use it initially in **one stage only** (recommended: `color_adjust` or `drc_plus_color`).
- Default config keeps it **off**, and even when enabled uses gentle deltas.

## What’s missing vs a “full ISP”
mini-ISP v0.1 includes a realistic single-frame skeleton, but does not yet include:
- MFNR / HDR merge (multi-frame pipeline containers)
- advanced artifact controls (false color / moiré suppression, de-ring)
- explicit gamut mapping / richer look LUT stages
- closed-loop AE/AWB/AF control (stats exist; control loop is out of scope for v0.1)
