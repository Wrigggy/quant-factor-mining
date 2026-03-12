# Methodology

## Objective
Build a reproducible multi-factor research pipeline for rigorous experimentation with strict anti-leakage controls.

## Core Workflow
1. Load snapshot market data (or refresh live data into snapshot).
2. Preprocess and validate data contract: index `(date, ticker)`, OHLCV columns, sorted unique index.
3. Compute standardized factor signals (momentum, mean-reversion, low-volatility).
4. Compute forward returns labels.
5. Run walk-forward evaluation:
   - Train window estimates factor weights by train-period IC means.
   - Test window applies fixed train weights only.
   - Backtest uses signal-at-`t` and execute-from-`t+1` timing.
6. Optionally reserve untouched final holdout period and evaluate once after selection.
7. Resolve benchmark series (SPY preferred, equal-weight fallback) and compute benchmark-relative attribution for each fold (alpha, beta, tracking error, IR).
8. Optionally bootstrap confidence intervals for alpha annualized and information ratio.
9. Persist artifacts (metrics, fold table, equity curve, config snapshot, report).

## Leakage Controls
- No future data in train/test split.
- No same-day signal execution.
- No dynamic test-period refit inside each fold.

## Cost Model
- Baseline linear turnover cost in basis points.
- Optional liquidity/slippage model:
  - commission + half-spread base
  - impact linked to participation ratio
  - cap-breach penalty beyond participation limit
- Cost is applied on rebalance dates.

## Limitations
- Uses simple equal-weight top-N construction.
- Uses synthetic snapshot by default unless live mode is enabled.
- Not a production trading system.
