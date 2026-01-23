# Testing

This document describes how to test mini-ISP in a way that keeps results **reproducible** and changes **safe** as the codebase grows.

## v0.1 focus
- **Correctness:** shapes, dtypes, value ranges, required metadata keys
- **Integration:** pipeline produces a valid run folder + `manifest.json`
- **Light regression:** metrics drift stays within tolerance (where applicable)

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

## Test categories (planned / recommended)

### 1) Stage contract tests
**Goal:** ensure each stage obeys I/O contracts.

Examples:
- `tests/test_stage_contracts.py`
  - verifies input/output format names (e.g., `RAW_BAYER_F32 → RGB_LINEAR_F32`)
  - asserts output shape/channel order
  - asserts numeric sanity (no NaNs/inf; expected range)

### 2) Manifest + artifacts tests
**Goal:** ensure the viewer contract stays stable.

Examples:
- `tests/test_manifest.py`
  - validates required fields exist in `manifest.json`
  - checks referenced artifact paths exist
  - ensures stage ordering/prefixing is consistent

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
