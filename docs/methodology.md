# Methodology

## Objective
Build a reproducible multi-factor research pipeline for internship demonstration with strict anti-leakage controls.

## Core Workflow
1. Load snapshot market data (or refresh live data into snapshot).
2. Preprocess and validate data contract: index `(date, ticker)`, OHLCV columns, sorted unique index.
3. Compute standardized factor signals (momentum, mean-reversion, low-volatility).
4. Compute forward returns labels.
5. Run walk-forward evaluation:
   - Train window estimates factor weights by train-period IC means.
   - Test window applies fixed train weights only.
   - Backtest uses signal-at-`t` and execute-from-`t+1` timing.
6. Compute benchmark-relative attribution for each fold (alpha, beta, tracking error, IR).
7. Persist artifacts (metrics, fold table, equity curve, config snapshot, report).

## Leakage Controls
- No future data in train/test split.
- No same-day signal execution.
- No dynamic test-period refit inside each fold.

## Cost Model
- Linear turnover cost in basis points.
- Cost applied on rebalance dates.

## Limitations
- Uses simple equal-weight top-N construction.
- Uses synthetic snapshot by default unless live mode is enabled.
- Not a production trading system.
