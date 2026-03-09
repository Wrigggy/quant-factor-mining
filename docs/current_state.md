# Current State Baseline (Migration Snapshot)

Date: 2026-03-09

## Legacy Issues Observed Before Migration

1. Root-level pytest files mixed script-style tests and missing fixtures.
2. Live network dependency caused fragile test execution.
3. Backtest and regime flows had timing assumptions requiring explicit review.
4. Reporting claims and test reliability were not aligned.

## Migration Direction

- Added new `src/qfm` architecture with strict data contracts.
- Added offline deterministic tests under `tests/`.
- Added script-based reproducible workflow writing run artifacts.

Legacy modules remain in place for backward compatibility, while new workflows use `src/qfm`.
