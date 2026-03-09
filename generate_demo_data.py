"""Generate dashboard demo artifacts using the offline-first qfm pipeline.

Outputs:
- outputs/factor_monitor.db
- outputs/weight_history.parquet
- outputs/regime_history.parquet
- outputs/backtests/equity_curve.parquet
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qfm.backtest.engine import run_backtest_from_scores
from qfm.data.fetch import DEFAULT_SNAPSHOT_PATH, generate_synthetic_market_data, load_snapshot, save_snapshot
from qfm.data.preprocess import close_prices_wide, preprocess_market_data
from qfm.factors.mean_reversion import MeanReversionFactor
from qfm.factors.momentum import MomentumFactor
from qfm.factors.volatility import LowVolatilityFactor
from qfm.labels.forward_returns import compute_forward_returns
from qfm.modeling.feature_builder import build_factor_matrix


OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
(OUT / "backtests").mkdir(exist_ok=True)


def _load_or_create_snapshot() -> pd.DataFrame:
    path = ROOT / DEFAULT_SNAPSHOT_PATH
    if path.exists():
        return load_snapshot(path)
    data = generate_synthetic_market_data()
    save_snapshot(data, path)
    return data


def _detect_regime(market_returns: pd.Series, lookback: int = 63) -> pd.DataFrame:
    records = []
    rolling_vol = market_returns.rolling(lookback).std() * np.sqrt(252)
    for date in market_returns.index:
        hist = market_returns.loc[:date]
        if len(hist) < lookback:
            regime = "neutral"
        else:
            recent = hist.tail(lookback)
            autocorr = recent.autocorr(lag=1)
            vol = float(recent.std() * np.sqrt(252))
            vol_percentile = float((rolling_vol.loc[:date] < vol).mean())
            if vol_percentile > 0.8:
                regime = "high_vol"
            elif autocorr > 0.15:
                regime = "trending"
            elif autocorr < -0.15:
                regime = "mean_reverting"
            else:
                regime = "neutral"
        records.append({"date": date, "regime": regime})

    return pd.DataFrame(records).set_index("date")


def _rolling_ic(series: pd.Series, fwd_returns: pd.Series, window: int = 63) -> pd.DataFrame:
    aligned = pd.DataFrame({"factor": series, "fwd": fwd_returns}).dropna()
    daily_ic = aligned.groupby(level="date").apply(
        lambda frame: frame["factor"].corr(frame["fwd"], method="spearman")
    )
    rolling_mean = daily_ic.rolling(window).mean()
    rolling_std = daily_ic.rolling(window).std()
    rolling_icir = rolling_mean / (rolling_std + 1e-12)
    return pd.DataFrame({"ic_mean": rolling_mean, "icir": rolling_icir})


def main() -> None:
    data = _load_or_create_snapshot()
    clean, _ = preprocess_market_data(data)

    factors = [
        MomentumFactor(lookback=252, skip=21, name="momentum"),
        MeanReversionFactor(lookback=21, name="mean_reversion"),
        LowVolatilityFactor(window=63, name="low_volatility"),
    ]
    matrix = build_factor_matrix(clean, factors, normalize=True)
    fwd = compute_forward_returns(clean, horizon=21)

    # Build adaptive but deterministic weights from rolling IC means.
    ic_tables = {}
    for col in matrix.columns:
        ic_tables[col] = _rolling_ic(matrix[col], fwd, window=63)

    dates = sorted(matrix.index.get_level_values("date").unique())
    weight_rows = []
    monitor_rows = []

    for date in dates:
        ic_mean = pd.Series(
            {
                name: float(ic_tables[name].loc[date, "ic_mean"])
                if date in ic_tables[name].index
                else np.nan
                for name in matrix.columns
            }
        )
        icir = pd.Series(
            {
                name: float(ic_tables[name].loc[date, "icir"])
                if date in ic_tables[name].index
                else np.nan
                for name in matrix.columns
            }
        )

        pos = ic_mean.fillna(0.0).clip(lower=0.0)
        if pos.sum() <= 0:
            w = pd.Series(1.0 / len(matrix.columns), index=matrix.columns)
        else:
            w = pos / pos.sum()

        hhi = float((w**2).sum())
        weight_rows.append({"date": date, **w.to_dict()})

        for factor_name in matrix.columns:
            monitor_rows.append(
                {
                    "recorded_at": pd.Timestamp.utcnow().isoformat(),
                    "date": str(date),
                    "factor_name": factor_name,
                    "icir": float(icir[factor_name]) if pd.notna(icir[factor_name]) else None,
                    "ic_mean": float(ic_mean[factor_name]) if pd.notna(ic_mean[factor_name]) else None,
                    "hhi": hhi,
                    "skip_count": 0,
                }
            )

    weight_df = pd.DataFrame(weight_rows).set_index("date")
    weight_df.to_parquet(OUT / "weight_history.parquet")

    # Save monitor SQLite expected by dashboard.
    db_path = OUT / "factor_monitor.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS factor_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                date TEXT,
                factor_name TEXT NOT NULL,
                icir REAL,
                ic_mean REAL,
                hhi REAL,
                skip_count INTEGER
            )
            """
        )
        conn.execute("DELETE FROM factor_metrics")
        conn.executemany(
            """
            INSERT INTO factor_metrics (recorded_at, date, factor_name, icir, ic_mean, hhi, skip_count)
            VALUES (:recorded_at, :date, :factor_name, :icir, :ic_mean, :hhi, :skip_count)
            """,
            monitor_rows,
        )
        conn.commit()

    # Regime history
    close = close_prices_wide(clean)
    market_ret = np.log(close / close.shift(1)).mean(axis=1).dropna()
    regime_df = _detect_regime(market_ret)
    regime_df.to_parquet(OUT / "regime_history.parquet")

    # Backtest using simple composite score
    scores = matrix.mean(axis=1).unstack("ticker").reindex(close.index).sort_index()
    asset_returns = close.pct_change().fillna(0.0)
    bt = run_backtest_from_scores(
        scores=scores,
        asset_returns=asset_returns,
        top_n=max(1, len(close.columns) // 3),
        rebalance_frequency=21,
        transaction_cost_bps=10.0,
        initial_capital=1_000_000,
    )
    bt = bt.rename(columns={"portfolio_return": "daily_return"})
    bt[["portfolio_value", "daily_return"]].to_parquet(OUT / "backtests" / "equity_curve.parquet")

    print("Dashboard demo data generated successfully.")
    print(f"- {OUT / 'factor_monitor.db'}")
    print(f"- {OUT / 'weight_history.parquet'}")
    print(f"- {OUT / 'regime_history.parquet'}")
    print(f"- {OUT / 'backtests' / 'equity_curve.parquet'}")


if __name__ == "__main__":
    main()
