"""Momentum factor implementations."""

from __future__ import annotations

import pandas as pd

from .base import BaseFactor, safe_stack


class MomentumFactor(BaseFactor):
    """12-1 momentum style factor: return from t-skip-lookback to t-skip."""

    def __init__(self, lookback: int = 252, skip: int = 21, name: str = "momentum"):
        super().__init__(name=name)
        self.lookback = lookback
        self.skip = skip

    def compute_raw(self, market_data: pd.DataFrame) -> pd.Series:
        close = market_data["close"].unstack("ticker")
        score = close.shift(self.skip) / close.shift(self.lookback + self.skip) - 1
        return safe_stack(score).rename(self.name)
