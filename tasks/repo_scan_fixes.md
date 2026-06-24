# Repository Scan Fixes

## Goal

Apply the follow-up fixes from the repository scan on the isolated `repo_scan`
branch, without touching the dirty `codex/calibration_uncertainties` checkout.

## Approaches

1. Graceful fallbacks for optional or missing modules.
2. Direct bug fixes for confirmed runtime errors.
3. Documentation cleanup for references to absent files.

## Tasks

- [x] Add graceful missing-module behavior for advertised optional APIs.
- [x] Fix single-detector `LogLike` shape indexing.
- [x] Fix odd-length `nfft`/`infft` round-trip.
- [x] Fix Eryn chain cleaning for differing NaN masks.
- [x] Ensure LVK pocoMC output directory exists.
- [x] Remove README/docs/example references to absent files.
- [x] Add minimal tests for changed behavior.
- [x] Run focused and full verification.

## Status

Completed on `repo_scan`.

## Verification

- `python -m compileall -q src tests examples`: passed.
- `.venv/bin/ruff check` on changed Python/test files: passed.
- `rg` search for removed stale paths: no matches.
- `.venv/bin/python -m pytest`: 28 passed, 2 skipped.

## Notes

- `ruff format --check` still wants broad formatting in legacy files, especially
  `src/hyperwave/inference/sampling.py`; I did not reformat those files to avoid
  unrelated churn.
- A separate detector-dependent-noise diff appeared repeatedly in the temporary
  worktree while working. It was left unstaged because it is unrelated to this
  request.
