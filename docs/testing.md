# Testing

This document describes how to test mini-ISP in a way that keeps results **reproducible** and changes **safe** as the codebase grows.

## Current focus
- **Correctness:** shapes, dtypes, value ranges, required metadata keys
- **Integration:** pipeline produces a valid run folder + `manifest.json`
- **Light regression:** basic checks for deterministic outputs (where applicable)

---

## Setup

Assumes you’ve followed the README Quickstart (venv + runtime deps). Then install test deps:

```bash
pip install -r requirements-dev.txt
# or minimally:
pip install pytest
```

---

## Run tests

```bash
pytest -q
```

Useful variants:

```bash
pytest -q -k manifest          # run only tests matching "manifest"
pytest -q -k stage_contracts   # contract tests
pytest -q -x                   # stop on first failure
```

---

## Test categories (current + planned)

### 1) Stage contract tests
**Goal:** ensure each stage obeys I/O contracts.

Examples (current):
- `tests/test_wb_demosaic.py`, `tests/test_dpc_lsc.py`, `tests/test_denoise_jdd.py`, `tests/test_ccm_stats.py`
  - verify shape/dtype, ranges, and required keys per stage

Planned:
- `tests/test_stage_contracts.py` (formalized contract checks)

### 2) Manifest + artifacts tests
**Goal:** ensure the viewer contract stays stable.

Examples (current):
- `tests/test_smoke.py`
  - validates run folder, `manifest.json`, and stage artifacts exist

Planned:
- `tests/test_manifest.py` (manifest schema-focused checks)

### 3) End-to-end smoke tests
**Goal:** prevent “it doesn’t run” regressions.

Examples:
- `tests/test_smoke.py`
  - runs `pipeline_mode: classic` on a tiny sample input
  - asserts run folder exists and has `final/output.png` and stage folders

### 4) Regression tests (light)
**Goal:** catch large behavior changes without overfitting.

Examples:
- check stage timing is present (not exact time)
- compare simple statistics (mean luminance, histogram bins) within tolerances
- keep tolerances wide in early versions; tighten only when stable

---

## Test data

Recommended structure:
- `tests/data/` for tiny fixtures (very small RAW-like arrays, or small PNGs)
- avoid large proprietary RAW files in the repo
- keep deterministic seeds for any randomized operations

---

## Notes
- Prefer **unit tests** around stage functions over heavy end-to-end tests.
- Treat the **manifest schema** as a compatibility contract: tests should protect it.
