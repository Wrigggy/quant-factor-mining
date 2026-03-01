# Tier 3 Design — Parameter Optimization, Notebooks, Dashboard

**Date:** 2026-03-02
**Scope:** Issues #9, #11, #12 (ML combination excluded)

---

## Issue #9 — Parameter Optimizer (Optuna)

**File:** `src/optimization/parameter_optimizer.py`

**Objective:** Maximize mean ICIR across walk-forward folds.

**Search space:**
| Parameter | Type | Range |
|---|---|---|
| `ic_window` | int | 21–126 |
| `decay_halflife` | int | 5–63 |
| `min_weight` | float | 0.01–0.20 |
| `normalization_method` | categorical | softmax, relu, rank |
| `forward_period` | int | 5–42 |

**Walk-forward:** 252-day train + 63-day test, rolled across history. Score = mean ICIR across all folds.

**Class:** `ParameterOptimizer(factors, data, prices, n_trials=100)`
- `optimize() → dict` — runs study, returns best params
- `get_study() → optuna.Study` — full study for inspection
- Saves best params to `outputs/optimization/best_params.json`

**New dependency:** `optuna`

---

## Issue #11 — Jupyter Notebooks (5)

**Style:** Explanatory with charts — markdown prose + code cells + matplotlib/seaborn plots. Runnable top-to-bottom with existing cached demo data.

| File | Content |
|---|---|
| `notebooks/02_factor_analysis.ipynb` | IC time series, IC decay curve, IC distribution per factor |
| `notebooks/03_regime_analysis.ipynb` | Regime state timeline, weight shifts by regime, frequency breakdown |
| `notebooks/04_portfolio_construction.ipynb` | Efficient frontier, weight evolution heatmap, turnover over time |
| `notebooks/05_backtest_results.ipynb` | Equity curve, monthly returns heatmap, drawdown, Sharpe/Sortino |
| `notebooks/06_live_trading_setup.ipynb` | Step-by-step guide: data → factors → monitor → trade (no charts) |

---

## Issue #12 — Streamlit Dashboard

**File:** `dashboard/app.py`

**Architecture:** Reads from pre-written output files and FactorMonitor SQLite DB. System runs separately; dashboard is a read-only viewer. Auto-refreshes every 30 seconds via `st.rerun()`.

**4 tabs:**
1. **Factor Health** — ICIR bar chart, HHI gauge, IC skip table (source: SQLite)
2. **Weight Evolution** — line chart per factor over time (source: `outputs/weight_history.parquet`)
3. **Regime State** — current regime badge + timeline (source: `outputs/regime_history.parquet`)
4. **Backtest** — equity curve, monthly returns heatmap (source: `outputs/backtests/`)

**Sidebar:** last-updated timestamp, refresh interval slider (10–120s), manual refresh button.

**Run command:** `streamlit run dashboard/app.py`

**New dependencies:** `streamlit`, `plotly`

---

## Files to Create/Modify

| Action | File |
|---|---|
| Create | `src/optimization/parameter_optimizer.py` |
| Create | `notebooks/02_factor_analysis.ipynb` |
| Create | `notebooks/03_regime_analysis.ipynb` |
| Create | `notebooks/04_portfolio_construction.ipynb` |
| Create | `notebooks/05_backtest_results.ipynb` |
| Create | `notebooks/06_live_trading_setup.ipynb` |
| Create | `dashboard/__init__.py` |
| Create | `dashboard/app.py` |
| Modify | `requirements.txt` (add optuna, streamlit, plotly) |
