# Interview Notes

## 30-Second Pitch
I built a leakage-safe, reproducible walk-forward multi-factor research pipeline with cost-aware backtesting and artifact tracking.

## Design Decisions You Can Defend
1. Offline-first reproducibility: tests and baseline runs do not depend on live API availability.
2. Explicit no-lookahead execution timing: signal at t, active from t+1.
3. Walk-forward validation instead of in-sample-only performance.
4. Artifacts per run: config snapshot, fold metrics, equity curve, report.

## Typical Interview Questions
- How did you prevent look-ahead bias?
- Why did you choose walk-forward over a single split?
- How sensitive are results to costs and rebalance frequency?
- What would be your next production-hardening step?

## Strong Follow-up Answers
- Add benchmark-relative attribution and risk decomposition.
- Add liquidity and slippage model calibration by ADV buckets.
- Add CI checks on leakage tests and schema contracts.
