# AGENTS.md (Central Manifest)

Scope: this file applies to the entire `mini-ISP` repository.

## Canonical instruction source
- This is the single source of truth for agent working rules.
- `CLAUDE.md` and `GEMINI.md` should only point here; do not duplicate policy text there.

## Project constraints
- Preserve run layout and schema compatibility:
  - do not break `runs/<run_id>/...` structure
  - do not break `manifest.json` schema
  - do not break viewer asset paths/contracts
- Keep dependencies minimal; do not add heavy runtime dependencies unless explicitly requested.
- RAW path target is Bayer mosaic workflow; unsupported non-mosaic RAW/DNG payloads should fail clearly.

## CCM policy guardrails
- Keep stage math (`ccm`) separate from resolver/defaulting policy in runner.
- Precedence must remain explicit:
  1) explicit `stages.ccm.*` config
  2) resolver auto-default (input-kind specific)
  3) deterministic identity fallback with recorded reason
- Record provenance in `stages/<nn>_ccm/debug.json` params for any auto-default path.

## Coding workflow expectations
- Fix root cause, avoid superficial patches.
- Keep changes focused; avoid unrelated refactors.
- For behavior changes, update docs in the same commit:
  - `README.md`
  - `docs/pipeline.md`
  - `docs/stage_contracts.md`
  - `docs/prompts.md` (spec/prompt tracking)

## Testing expectations
- Run targeted tests first, then full suite when practical:
  - preferred final check: `python -m pytest -q`
- If runtime is heavy for full RAW runs, use representative runs and document any performance tradeoffs.

## Commit hygiene
- Use atomic commits with aligned code + docs.
- Do not include changes from external validation repositories in `mini-ISP` commits unless explicitly requested.
