# Artifacts and Viewer

mini-ISP is designed to be **inspectable**. Each run produces a structured folder containing:
- per-stage outputs (images + JSON)
- a stable `manifest.json`
- a static HTML viewer that lets you scrub through stages

This document defines the **run folder layout**, **manifest schema**, and the **static viewer contract**.

---

## 1) Run folder layout

A run produces a directory like:

```
runs/<run_id>/
  manifest.json
  config_resolved.yaml
  final/
    output.png
  stages/
    00_raw_norm/
      preview.png
      roi.png
      debug.json
      timing_ms.json
    01_dpc/
      ...
    02_lsc/
      ...
    ...
  viewer/
    index.html
    app.js
    styles.css
```

Notes:
- Stage folders are ordered with a numeric prefix for stable sorting.
- `final/` contains the final display output (after `oetf_encode`).
- `config_resolved.yaml` is the exact config used for the run (after defaults/overrides).

---

## 2) Standard per-stage artifacts

When dumping is enabled, each stage should emit:

Required:
- `preview.png` — downscaled full-frame view (display-ready for the viewer)
- `debug.json` — parameters + key metrics + warnings
- `timing_ms.json` — elapsed time (ms) for the stage
Note: If ROI dumping is enabled, stages should write `roi.png` when meaningful.

Optional:
- `roi.png` — ROI crop preview when ROI dump enabled
- `hist.json` — histogram data (if enabled)
- `mask.png` — for mask-like stages (e.g., skin mask when enabled)
- `extra/` — any additional stage-specific outputs

---

## 3) `manifest.json` schema

The viewer is driven by `manifest.json`. Keep it **stable** as the project evolves.

### 3.1 Top-level shape
```json
{
  "schema_version": "0.1",
  "run_id": "2026-01-24_18-30-12",
  "title": "mini-ISP run",
  "input": {
    "path": "data/sample.dng",
    "width": 4032,
    "height": 3024,
    "cfa_pattern": "RGGB"
  },
  "pipeline_mode": "classic",
  "final": {
    "path": "final/output.png"
  },
  "stages": [
    {
      "index": 0,
      "name": "raw_norm",
      "display_name": "RAW normalize",
      "dir": "stages/00_raw_norm",
      "artifacts": {
        "preview": "stages/00_raw_norm/preview.png",
        "roi": "stages/00_raw_norm/roi.png",
        "debug": "stages/00_raw_norm/debug.json",
        "timing": "stages/00_raw_norm/timing_ms.json"
      },
      "timing_ms": 3.8
    }
  ]
}
```

### 3.2 Required fields
- `schema_version` (string): `"0.1"`
- `run_id` (string)
- `input.path` (string)
- `pipeline_mode` (string)
- `final.path` (string)
- `stages` (array)

Each stage entry must include:
- `index` (int, 0-based)
- `name` (string, canonical stage name)
- `dir` (string, stage folder)
- `artifacts.preview` (string)
- `artifacts.debug` (string)
- `artifacts.timing` (string)

### 3.3 Optional fields
- `title` (string)
- `input.width`, `input.height` (int)
- `input.cfa_pattern` (string)
- `input.bit_depth`, `input.black_level`, `input.white_level` (numbers; RAW only)
- `stages[].display_name` (string)
- `stages[].artifacts.roi` (string)
- `stages[].timing_ms` (number) optional convenience copy of the value in `timing_ms.json` for faster viewer display

---

## 4) `debug.json` conventions (recommended)

Each stage’s `debug.json` should be human-readable and include:
- `stage`: stage name
- `params`: resolved parameters
- `metrics`: key numeric metrics
- `warnings`: list of strings
- `notes`: list of strings (optional)

Example:
```json
{
  "stage": "dpc",
  "params": { "method": "median", "threshold": 0.25 },
  "metrics": { "n_fixed": 1240 },
  "warnings": [],
  "notes": []
}
```

---

## 5) Static viewer contract

The viewer is a static page that:
- loads `manifest.json` (single-run mode)
- can load a compare bundle (A/B compare mode) and show side-by-side previews
- shows the stage list in order
- displays `preview.png` (and ROI when available)
- shows the `debug.json` content and timing
Note: open the viewer via HTTP (e.g., `python -m http.server`) since `file://` may block `fetch()` in some browsers.

### 5.1 What the viewer must not assume
- It must not assume every stage has ROI/mask/hist artifacts.
- It must not assume stage names beyond what appears in `manifest.json`.

### 5.2 Minimal interactions
- select stage from list (or left/right arrow)
- toggle full-frame vs ROI (if ROI exists)
- show/hide debug JSON panel

---

## 6) Extending the viewer later (planned)

The static viewer is designed to upgrade without breaking old runs:
- add histogram charts (read `hist.json`)
- add overlays (mask heatmap on preview)
- move to a React UI later while keeping `manifest.json` stable

See: `docs/roadmap.md`
