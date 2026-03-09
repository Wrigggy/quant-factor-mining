"""Volatility factor implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseFactor, safe_stack


class LowVolatilityFactor(BaseFactor):
    """Low-vol anomaly proxy: negative rolling volatility."""

    def __init__(self, window: int = 63, annualize: bool = True, name: str = "low_volatility"):
        super().__init__(name=name)
        self.window = window
        self.annualize = annualize

    def compute_raw(self, market_data: pd.DataFrame) -> pd.Series:
        close = market_data["close"].unstack("ticker")
        returns = close.pct_change()
        vol = returns.rolling(self.window, min_periods=max(5, self.window // 3)).std()
        if self.annualize:
            vol = vol * np.sqrt(252)
        score = -vol
        return safe_stack(score).rename(self.name)
