# Quant Factor Mining Research Project

A leakage-safe, reproducible multi-factor research framework for equity factor signals.

This project is designed for rigorous, reproducible factor research:
- strict time-ordering in research and backtest
- offline reproducibility by default
- run artifacts saved per experiment

## What This Project Demonstrates

1. Factor research with explicit no-lookahead controls.
2. Walk-forward validation (train/test rolling windows).
3. Cost-aware portfolio backtesting.
4. Benchmark-relative attribution (alpha/beta/tracking error/information ratio).
5. Bootstrap confidence intervals for attribution (alpha/IR).
6. Nested selection with untouched holdout evaluation.
7. Reproducible outputs (metrics, equity curve, config snapshot, report).

## Repository Structure

```text
src/qfm/
  data/         # contracts, preprocessing, snapshot/live fetch
  factors/      # momentum, mean-reversion, low-volatility
  labels/       # forward return labels
  modeling/     # feature build, walk-forward, parameter search
  portfolio/    # risk model + optimizer helpers
  backtest/     # engine, costs, metrics
  reporting/    # report generation

configs/
  base.yaml
  strategy/default.yaml

scripts/
  refresh_data.py
  run_walkforward.py
  run_backtest.py
  run_parameter_search.py
  generate_report.py

tests/
  unit/
  integration/

artifacts/runs/<timestamp>/
  config_snapshot.yaml
  run_snapshot.json
  fold_metrics.csv
  equity_curve.parquet
  metrics.json
  holdout_metrics.json
  holdout_equity_curve.parquet
  nested_search_results.csv
  stability_scoreboard.csv
  stability_summary.json
  selected_params.json
  report.md

legacy/
  # archived pre-migration modules, scripts, and notes
```

## Quick Start

### 1. Install

```bash
cd /Users/kevinwu/Coding/quant-factor-mining
pip install -r requirements.txt
```

### 2. Run Offline Snapshot Workflow (Recommended)

```bash
# Create or refresh local snapshot (synthetic deterministic data)
python3 scripts/refresh_data.py --config configs/base.yaml

# Run full walk-forward research
python3 scripts/run_walkforward.py --config configs/strategy/default.yaml

# Generate report for latest run
python3 scripts/generate_report.py
```

### 3. Run Single Full-Period Backtest

```bash
python3 scripts/run_backtest.py --config configs/strategy/default.yaml
```

### 4. Optional Live Data Refresh

```bash
python3 scripts/refresh_data.py --config configs/base.yaml --live
python3 scripts/run_walkforward.py --config configs/strategy/default.yaml --live
```

### 5. Parameter Search (Deterministic Grid)

```bash
python3 scripts/run_parameter_search.py --config configs/strategy/default.yaml
```

### 6. Stability-First Selection (Holdout Gate)

```bash
python3 scripts/run_walkforward.py --config configs/strategy/stability.yaml
python3 scripts/run_parameter_search.py --config configs/strategy/stability.yaml --selection-mode stability_first
```

## Testing

All tests are offline and deterministic.

```bash
python3 -m pytest -q
```

## Methodology Notes

- Signal at `t` is never applied to return at `t`.
- Rebalance occurs at close of `t`; new weights are active from `t+1`.
- Walk-forward folds estimate factor weights on train only, then evaluate on test.
- Fold shift is configurable via `research.walkforward_step_size`.
- Nested search ranks parameter sets on pre-holdout folds only.
- Optional holdout window is untouched during parameter selection.
- Position sizing can be equal-weight or score-tilted with per-name cap.
- Attribution metrics can include bootstrap confidence intervals.
- Cost model supports both linear bps and liquidity/slippage-aware mode.
- Benchmark defaults to SPY when available, with equal-weight fallback for offline reliability.

See:
- `docs/methodology.md`
- `docs/model_equations.md`
- `docs/assumptions.md`
- `docs/limitations.md`
- `docs/project_notes.md`

## Scope and Limitations

- This is a research framework, not production trading software.
- Default mode uses synthetic snapshot for reproducibility.
- Cost model remains a proxy (not LOB simulation), even with liquidity/slippage extensions.
