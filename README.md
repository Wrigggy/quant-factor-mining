# Quant Factor Mining

A leakage-safe, walk-forward multi-factor research framework for equity signals. Built for rigorous, reproducible factor research with strict time-ordering, cost-aware backtesting, and holdout validation.

## Overview

This framework implements a complete factor research pipeline:

```
OHLCV Data → Factor Computation → Walk-Forward Validation → Portfolio Backtest → Attribution & Reporting
```

**Three classic equity factors** are computed, cross-sectionally z-scored, and combined via information-coefficient (IC) weighting:

| Factor | Formula | Default Params | Intuition |
|--------|---------|---------------|-----------|
| **Momentum** | `close(t−skip) / close(t−lookback−skip) − 1` | lookback=252d, skip=21d | Medium-term continuation |
| **Mean Reversion** | `−(close(t) / close(t−lookback) − 1)` | lookback=21d | Short-term reversal |
| **Low Volatility** | `−std(returns, window) × √252` | window=63d | Lower risk, higher score |

**Key design principles:**
- Signal at `t` is applied from `t+1` — no lookahead leakage
- Walk-forward folds estimate weights on train data only
- Holdout period is untouched during parameter selection
- Deterministic synthetic data by default (seed=42) for full reproducibility

## Project Structure

```
src/qfm/
├── data/           Data contracts, preprocessing, snapshot & live fetch
├── factors/        Momentum, mean-reversion, low-volatility implementations
├── labels/         Forward return label computation
├── modeling/       Walk-forward engine, feature builder, parameter search, stability
├── portfolio/      Risk model, mean-variance optimizer, constraints
├── backtest/       Backtest engine, transaction costs, performance metrics, benchmarks
└── reporting/      Tearsheet & table generation

configs/
├── base.yaml                 Data config (10 tickers, 2018–2024)
└── strategy/
    ├── default.yaml          Full strategy with nested parameter search
    └── stability.yaml        Stability-first selection with holdout gate

scripts/
├── refresh_data.py           Generate synthetic or fetch live market data
├── run_walkforward.py        Main walk-forward research pipeline
├── run_backtest.py           Single full-period backtest
├── run_parameter_search.py   Grid search over factor parameters
└── generate_report.py        Generate markdown report from run artifacts

tests/
├── unit/                     17 test files covering contracts, factors, sizing, costs
└── integration/              Full pipeline & holdout split tests

dashboard/                    Streamlit factor monitor (read-only)
docs/                         Methodology, equations, assumptions, limitations
artifacts/runs/<timestamp>/   Per-run output directory
```

## Quick Start

### Install

```bash
git clone https://github.com/Wrigggy/quant-factor-mining.git
cd quant-factor-mining
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# 1. Generate deterministic snapshot data
python3 scripts/refresh_data.py --config configs/base.yaml

# 2. Run walk-forward research with nested parameter search
python3 scripts/run_walkforward.py --config configs/strategy/default.yaml

# 3. Generate human-readable report
python3 scripts/generate_report.py
```

Results are saved to `artifacts/runs/<timestamp>/` with full config snapshots for reproducibility.

### Other Workflows

```bash
# Single full-period backtest
python3 scripts/run_backtest.py --config configs/strategy/default.yaml

# Standalone grid search
python3 scripts/run_parameter_search.py --config configs/strategy/default.yaml

# Stability-first selection with holdout gating
python3 scripts/run_walkforward.py --config configs/strategy/stability.yaml

# Use live market data from Yahoo Finance
python3 scripts/refresh_data.py --config configs/base.yaml --live
python3 scripts/run_walkforward.py --config configs/strategy/default.yaml --live
```

### Demo Dashboard

A Streamlit dashboard renders pre-computed factor weights, regime history, and
the equity curve from `outputs/`. Use this for a quick visual walkthrough.

```bash
# (one-time) build dashboard artifacts under outputs/
python3 generate_demo_data.py

# launch the dashboard (defaults to http://localhost:8501)
streamlit run dashboard/app.py
```

If `outputs/factor_monitor.db`, `outputs/weight_history.parquet`,
`outputs/regime_history.parquet`, and `outputs/backtests/equity_curve.parquet`
already exist, you can skip `generate_demo_data.py` and launch the dashboard
directly.

Approximate timings (synthetic data, 10 tickers, 2018–2024):

| Step | Wall-clock |
|------|------------|
| `generate_demo_data.py` (rebuild artifacts) | ~20–60 s |
| `streamlit run dashboard/app.py` (artifacts cached) | ~5–10 s to first paint |
| `scripts/refresh_data.py` | a few seconds |
| `scripts/run_walkforward.py` (nested search, default config) | a few minutes |
| `scripts/generate_report.py` | seconds |

Quick demo path (~1 min): launch the dashboard against cached artifacts.
Full methodology demo (~3–5 min): `refresh_data` → `run_walkforward` →
`generate_report`.

## Pipeline Details

### Data

- **Default:** Synthetic OHLCV for 10 large-cap US equities (AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, JPM, XOM, JNJ) over 2018–2024
- **Format:** MultiIndex `(date, ticker)` DataFrame with OHLCV columns, validated by a strict data contract
- **Preprocessing:** Forward-fill gaps (max 3 days), remove invalid bars (high < low)
- **Live mode:** Fetches real data via `yfinance` with `--live` flag

### Walk-Forward Validation

Rolling train/test splits with configurable window sizes:
- **Train window:** 504 days (~2 years) — used to estimate IC-based factor weights
- **Test window:** 126 days (~6 months) — out-of-sample evaluation
- **Step size:** Configurable (default = test size, or 21 days for granular analysis)

Factor weights are the Spearman IC of each factor against forward returns, estimated on train data only. Composite score = IC-weighted sum of normalized factors.

### Backtesting

- **Position sizing:** Equal-weight or score-tilted (softmax with temperature + per-name weight cap)
- **Rebalancing:** Every N days (default 21), with turnover tracking
- **Cost models:** Linear bps or full liquidity model (commission + spread + slippage + market impact)
- **Benchmark:** SPY when available, equal-weight fallback for offline mode

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted return (annualized) |
| Alpha | Excess return vs. benchmark (annual) |
| Beta | Sensitivity to benchmark |
| Information Ratio | Active return / tracking error |
| Max Drawdown | Worst peak-to-trough decline |
| Bootstrap CIs | 95% confidence intervals for alpha and IR |

### Holdout Gating

An untouched holdout period (default: last 126 days) is reserved for final validation. The holdout gate checks minimum Sharpe and excess return thresholds — parameter sets that fail are flagged, preventing overfitted configs from passing selection.

## Run Artifacts

Each run produces a timestamped directory under `artifacts/runs/` containing:

| File | Description |
|------|-------------|
| `metrics.json` | Aggregate metrics across all folds |
| `fold_metrics.csv` | Per-fold performance breakdown |
| `equity_curve.parquet` | Daily portfolio value time series |
| `holdout_metrics.json` | Untouched holdout period evaluation |
| `nested_search_results.csv` | Grid search results ranked by selection metric |
| `selected_params.json` | Best parameter set from search |
| `stability_summary.json` | Fold dispersion diagnostics |
| `config_snapshot.yaml` | Full config used for the run |
| `run_snapshot.json` | Run metadata (data summary, timestamp) |
| `report.md` | Human-readable markdown report |

## Testing

All tests are offline and deterministic:

```bash
python3 -m pytest -q
```

Coverage includes data contract validation, no-lookahead verification, factor index hygiene, position sizing, benchmark attribution, liquidity costs, walk-forward fold generation, and stability selection.

## Documentation

- [`docs/methodology.md`](docs/methodology.md) — Core workflow and leakage controls
- [`docs/model_equations.md`](docs/model_equations.md) — Full mathematical specification
- [`docs/assumptions.md`](docs/assumptions.md) — Key assumptions
- [`docs/limitations.md`](docs/limitations.md) — Known limitations

## Limitations

- This is a **research framework**, not production trading software
- Default mode uses synthetic data — factor premia are not embedded in the generator, so performance reflects noise characteristics rather than real market anomalies
- Cost model is a proxy (not LOB simulation), even with liquidity extensions
- Factor universe is limited to three classic signals; extending requires subclassing `BaseFactor`