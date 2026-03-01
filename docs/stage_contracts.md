# Stage Contracts

This document defines the **interfaces (“contracts”)** between stages in mini-ISP. The goal is that stages are **swappable**, pipeline modes are **comparable**, and debugging artifacts are **consistent**.

> TL;DR: Each stage consumes a `Frame` and returns a `Frame` with a well-defined `image` format + required metadata. Stages may emit optional artifacts, but should not silently change data conventions.

---

## 1. Core concepts

### 1.1 Frame
A `Frame` is the unit passed between stages:

- `image`: ndarray (either RAW mosaic or RGB)
- `meta`: dict (camera + processing metadata)

A stage must:
- **not mutate** its input `Frame` in-place (copy or create new output)
- declare its **input format** and **output format**
- write deterministic artifacts when `dump.enable: true`

### 1.2 Image formats
We keep the format set intentionally small. Use these canonical format names in code and manifests:

- `RAW_BAYER_F32`  
  H×W float32 mosaic, normalized nominally to [0, 1] after `raw_norm`
  (non-mosaic/RGB DNG payloads are out of scope for this path)

- `RGB_LINEAR_F32`  
  H×W×3 float32, linear light, nominally [0, 1] (values may exceed slightly before clipping)

- `RGB_DISPLAY_U8`  
  H×W×3 uint8, encoded for display (default sRGB OETF)

> Note: “display output” is the concept; the default encoding is sRGB, but `oetf_encode` is designed to extend to other encodes/bit-depths later.

### 1.3 Coordinate conventions
- Image arrays are **row-major**: `image[y, x]`
- RGB channel order is **RGB**
- CFA pattern (for RAW) must be explicit in metadata: `meta.cfa_pattern ∈ {RGGB, BGGR, GRBG, GBRG}`

---

## 2. Metadata keys (minimum)

Stages may add keys, but must not delete required keys.

### 2.1 Required after RAW ingest
- `meta.source_path`: input filename
- `meta.bit_depth`: original sensor bit depth (if known)
- `meta.black_level`: scalar or per-channel (if known)
- `meta.white_level`: scalar (if known)
- `meta.cfa_pattern`: RGGB/BGGR/GRBG/GBRG (required for demosaic/JDD)
- `meta.camera_id`: optional string (if known)

### 2.2 Required after WB
- `meta.wb_gains`: `[r, g, b]` (applied to both green sites)
- `meta.wb_applied`: bool

### 2.3 Color transform metadata
- `meta.ccm`: 3×3 matrix used (even if identity)
- `meta.ccm_mode`: `identity|manual|profile|chain`

### 2.4 Stats metadata
- `meta.stats_3a`: dict with fields produced by `stats_3a` (see stage section)

### 2.5 Optional resolver metadata (RAW auto-default inputs)
- `meta.daylight_wb_gains`: optional `[r, g, b]` daylight WB triplet
- `meta.non_dng_cam_to_xyz_matrix`: optional non-DNG metadata-derived 3×3 camera→XYZ matrix
- `meta.non_dng_selected_input_variant`: optional non-DNG input-variant hint (for resolver use)

---

## 3. Standard artifacts (per stage)

When dumping is enabled, each stage should emit:

- `preview.png` — downscaled full-frame preview (displayed in viewer)
- `roi.png` — optional ROI crop preview (configurable ROI)
- `debug.json` — stage parameters + key metrics + any warnings
- `timing_ms.json` — elapsed time for this stage

Optional (stage-dependent):
- histograms, heatmaps, masks, patch metrics

---

## 4. Stage-by-stage contracts

### 4.1 `raw_norm`
**Input:** vendor RAW (reader-specific)  
**Output:** `RAW_BAYER_F32`

Responsibilities:
- apply black/white normalization (if metadata available)
- clamp or soft-clip to a sane range (configurable)
- populate `meta.black_level`, `meta.white_level` if known
- ensure `meta.cfa_pattern` is set (required downstream)
- validate RAW payload is a 2D Bayer mosaic; reject unsupported non-mosaic RAW/DNG inputs early

Artifacts:
- preview (simple debayer for visualization is ok, but do not change stored RAW)
- debug includes min/max/percentiles

---

### 4.2 `dpc`
**Input:** `RAW_BAYER_F32`  
**Output:** `RAW_BAYER_F32`

Responsibilities:
- correct dead/hot pixels (simple median-of-neighbors baseline)
- **same-CFA-only invariant**: when operating on a Bayer mosaic, any replacement value must be computed from neighbors at the **same CFA site** only (R from R, B from B, and treat the two green sites separately)
- report number of corrected pixels

Artifacts:
- optional defect map visualization
- debug: `n_fixed`, thresholds used, and neighbor policy/stat (e.g., `neighbor_policy="same_cfa_only"`, `neighbor_stat="median"`)

---

### 4.3 `lsc`
**Input:** `RAW_BAYER_F32`  
**Output:** `RAW_BAYER_F32`

Responsibilities:
- apply shading gain map (baseline radial model or provided map)
- avoid runaway gain (cap gain)
- optional toggle: if `params.enabled=false`, the runner bypasses LSC (no pixel change) while still emitting artifacts; debug reports it was skipped

Artifacts:
- gain map visualization (optional)
- debug: gain stats (min/max/mean), or when skipped: `skipped=true` (+ reason)

---

### 4.4 `wb_gains`
**Input:** `RAW_BAYER_F32`  
**Output:** `RAW_BAYER_F32`

Responsibilities:
- apply WB gains in RAW mosaic domain
- record `meta.wb_gains` and `meta.wb_applied = true`

Artifacts:
- debug: gains used

---

### 4.5 `demosaic`
**Input:** `RAW_BAYER_F32`  
**Output:** `RGB_LINEAR_F32`

Responsibilities:
- demosaic to linear RGB
- keep linear scale (no gamma here)

Artifacts:
- debug: method name + known artifact notes (zipper/false color)

---

### 4.6 `denoise`
**Input:** `RGB_LINEAR_F32`  
**Output:** `RGB_LINEAR_F32`

Responsibilities:
- baseline RGB denoise (e.g., gaussian/box/chroma_gaussian)
- preserve edges reasonably; avoid hue shifts

Artifacts:
- debug: strength parameters + optional noise estimate

---

### 4.7 `ccm`
**Input:** `RGB_LINEAR_F32`  
**Output:** `RGB_LINEAR_F32`

Responsibilities:
- apply 3×3 CCM in linear RGB
- treat CCM as camera-RGB → working/render-RGB transform (still linear RGB; not necessarily XYZ)
- support identity/manual/profile modes
- support `chain` mode (camera RGB → XYZ via `cam_to_xyz_matrix`, then XYZ → working via `xyz_to_working_matrix`, collapsed to an effective 3×3)
- default `xyz_to_working_matrix` is built-in XYZ(D65) → linear sRGB(D65); runner may inject other working transforms (for example D50-adapted paths)
- support runner-resolved auto-default policy without changing stage order:
  - DNG RAW auto-default path (v0.2-M8)
  - non-DNG RAW deterministic metadata path (v0.2-M11, policy id `non_dng_meta_default`)
    - matrix-source extraction refinement (v0.2-M12): deterministic `wp_error_min_det` selection from `rawpy.rgb_xyz_matrix` candidate interpretations
    - current rule: `wp_infer_clean_d65_d50_else_daylight_with_outlier_identity`
    - clean D65 -> `selected_input|d65`
    - clean D50 -> `selected_input|d50adapt`
    - ambiguous + daylight WB -> `pre_unwb_daylight|d65`
    - outlier high-error confidence fallback (`min(wp_err_d50, wp_err_d65) > 0.33`) -> identity (skip auto chain)
    - else fallback -> `pre_unwb_daylight|d65` (or `selected_input|d50adapt` if daylight WB unavailable)
- record `meta.ccm` and `meta.ccm_mode`

Artifacts:
- debug: mode + matrix (for `chain`: component matrices + effective matrix + source/provenance)
- when runner auto-default policy is active, debug params should include `auto_default_applied` and `auto_default_reason`
- for non-DNG auto-default, debug params should also include:
  - `ccm_source=non_dng_meta_default`
  - `non_dng_meta_rule`
  - `non_dng_meta_input_variant`
  - `non_dng_meta_wp_variant`
  - `non_dng_meta_branch`
  - `non_dng_meta_selection_reason`
  - `non_dng_meta_wp_err_d50`
  - `non_dng_meta_wp_err_d65`
  - `non_dng_meta_outlier_confidence_threshold`
  - `non_dng_meta_outlier_confidence_trigger`
  - `non_dng_meta_outlier_fallback_applied`
  - `non_dng_matrix_source_policy`
  - `non_dng_matrix_selected_source_variant`
  - `non_dng_matrix_selected_wp_err_d50`
  - `non_dng_matrix_selected_wp_err_d65`
  - `non_dng_matrix_selected_wp_err_min`
  - `non_dng_matrix_candidate_count`

---

### 4.8 `stats_3a`
**Input:** `RGB_LINEAR_F32` (after `ccm`) 
**Output:** `RGB_LINEAR_F32` (unchanged)

Responsibilities:
- compute and export non-invasive stats:
  - **AE**: luma histogram, mean/percentiles, highlight clipping %
  - **AWB**: gray-world / white-patch candidate stats (e.g., channel means in neutral candidates)
  - **AF**: focus metric (e.g., Tenengrad/variance of Laplacian) on configurable ROI
- write stats into `meta.stats_3a`

Artifacts:
- debug: selected ROI, metric values

---

### 4.9 `tone`
**Input:** `RGB_LINEAR_F32`  
**Output:** `RGB_LINEAR_F32`

Responsibilities:
- tone mapping / DRC in a defined domain
- current methods: `reinhard`, `filmic`
- future: `ai_drc` method (AI-DRC; model predicts curve/LUT/coeffs)

Artifacts:
- debug: curve params, clipping ratio before/after

---

### 4.10 `color_adjust`
**Input:** `RGB_LINEAR_F32`  
**Output:** `RGB_LINEAR_F32`

Responsibilities:
- small post-tone color adjustments (identity by default)
- examples: saturation roll-off, hue-preserve saturation, LUT-like mapping (later)

Artifacts:
- debug: enabled flags, strength params

---

### 4.11 `sharpen`
**Input:** `RGB_LINEAR_F32`  
**Output:** `RGB_LINEAR_F32`

Responsibilities:
- apply sharpening (unsharp mask baseline)
- minimize halos and ringing (basic thresholds)

Artifacts:
- debug: radius/amount/threshold

---

### 4.12 `oetf_encode`
**Input:** `RGB_LINEAR_F32`  
**Output:** `RGB_DISPLAY_U8` (default)

Responsibilities:
- apply output transfer (default sRGB OETF)
- quantize and save output image
- future: allow other encodings/bit-depths and non-RGB outputs (e.g., YUV)

Artifacts:
- final output image path
- debug: encoding, bit depth, clamp behavior

---

## 5. Composite stage contracts

### 5.1 `jdd_raw2rgb`
**Replaces:** `demosaic` + `denoise`  
**Input:** `RAW_BAYER_F32`  
**Output:** `RGB_LINEAR_F32`

Responsibilities:
- perform joint demosaic+denoise behavior
- must match I/O expectations of downstream stages

Artifacts:
- debug: method + strength knobs + artifact notes

---

### 5.2 `drc_plus_color`
**Replaces:** `tone` + `color_adjust`  
**Input:** `RGB_LINEAR_F32`  
**Output:** `RGB_LINEAR_F32`

Responsibilities:
- implement a curated DRC + color adjustment path
- keep parameters explicit so results are explainable and comparable

Artifacts:
- debug: DRC parameters + color adjustment parameters

---

## 6. Skin-mask hook (optional)

The skin mask is a **side signal**: it should not change the `Frame.image` format.

### Methods
- `heuristic`: fast heuristic mask (e.g., YCbCr/HSV thresholds + morphology)
- `seg_model`: semantic segmentation model (future)

Stages may opt-in to using the mask (recommended: `color_adjust` only), and must:
- behave sensibly when no mask is present
- keep changes gentle and easy to disable

Artifacts:
- `skin_mask.png` visualization (when enabled)
- debug: coverage %, method used
