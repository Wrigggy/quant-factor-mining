"""Data acquisition helpers with offline-first snapshot workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .contracts import ensure_market_data

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency path
    yf = None


DEFAULT_SNAPSHOT_PATH = Path("data/processed/us_equities_snapshot.parquet")


@dataclass(frozen=True)
class SnapshotMetadata:
    path: str
    rows: int
    n_tickers: int
    start_date: str
    end_date: str


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def generate_synthetic_market_data(
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    tickers: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV market data for offline tests.

    The synthetic process combines market, sector, and idiosyncratic components
    so factors have non-trivial but reproducible structure.
    """
    tickers = list(tickers or [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "XOM", "JNJ",
    ])

    dates = pd.bdate_range(start, end)
    n_days = len(dates)
    n_assets = len(tickers)

    rng = np.random.default_rng(seed)

    market = rng.normal(0.0002, 0.01, size=n_days)
    sector = rng.normal(0.0, 0.006, size=(n_days, 3))
    betas = rng.uniform(0.7, 1.3, size=n_assets)
    sector_loadings = rng.uniform(-0.4, 0.6, size=(n_assets, 3))
    idio = rng.normal(0.0, 0.012, size=(n_days, n_assets))

    sector_component = np.zeros((n_days, n_assets), dtype=float)
    for k in range(sector.shape[1]):
        sector_component += np.outer(sector[:, k], sector_loadings[:, k])

    returns = (
        market[:, None] * betas[None, :]
        + sector_component
        + idio
    )
    returns = np.clip(returns, -0.2, 0.2)

    close = 100 * np.exp(np.cumsum(returns, axis=0))
    gap = rng.normal(0.0, 0.003, size=(n_days, n_assets))
    open_px = close * np.exp(gap)
    intraday_spread = np.abs(rng.normal(0.0, 0.01, size=(n_days, n_assets)))
    high = np.maximum(open_px, close) * (1 + intraday_spread * 0.5)
    low = np.minimum(open_px, close) * (1 - intraday_spread * 0.5)
    volume = rng.integers(500_000, 8_000_000, size=(n_days, n_assets)).astype(float)

    arrays = [np.repeat(dates.values, n_assets), np.tile(np.array(tickers), n_days)]
    index = pd.MultiIndex.from_arrays(arrays, names=["date", "ticker"])

    frame = pd.DataFrame(
        {
            "open": open_px.reshape(-1),
            "high": high.reshape(-1),
            "low": low.reshape(-1),
            "close": close.reshape(-1),
            "volume": volume.reshape(-1),
        },
        index=index,
    )

    return ensure_market_data(frame)


def save_snapshot(data: pd.DataFrame, path: Path = DEFAULT_SNAPSHOT_PATH) -> SnapshotMetadata:
    """Save market data snapshot to parquet and return summary metadata."""
    market = ensure_market_data(data)
    _ensure_parent(path)
    market.to_parquet(path)

    dates = market.index.get_level_values("date")
    metadata = SnapshotMetadata(
        path=str(path),
        rows=int(len(market)),
        n_tickers=int(market.index.get_level_values("ticker").nunique()),
        start_date=str(dates.min().date()),
        end_date=str(dates.max().date()),
    )
    return metadata


def load_snapshot(path: Path = DEFAULT_SNAPSHOT_PATH) -> pd.DataFrame:
    """Load local snapshot; raises FileNotFoundError when missing."""
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    data = pd.read_parquet(path)
    return ensure_market_data(data)


def fetch_live_yfinance(
    tickers: Iterable[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch OHLCV using yfinance and return market-data-contract frame."""
    if yf is None:
        raise RuntimeError("yfinance is not available; install dependency or use snapshot mode")

    all_frames = []
    for ticker in tickers:
        history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if history.empty:
            continue
        history = history.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        history = history[["open", "high", "low", "close", "volume"]]
        history["ticker"] = str(ticker)
        history = history.reset_index().rename(columns={"Date": "date"})
        all_frames.append(history)

    if not all_frames:
        raise RuntimeError("No data fetched from yfinance")

    merged = pd.concat(all_frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.set_index(["date", "ticker"]).sort_index()

    return ensure_market_data(merged)
