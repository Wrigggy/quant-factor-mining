"""Factor base abstractions and shared utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from qfm.data.contracts import ensure_market_data


def safe_stack(frame: pd.DataFrame) -> pd.Series:
    """Compatibility stack helper across pandas versions."""
    try:
        stacked = frame.stack(future_stack=True)
    except TypeError:  # pragma: no cover - pandas<2.1 fallback
        stacked = frame.stack(dropna=False)
    except ValueError:
        # Fallback for pandas variants that reject future_stack argument combos.
        stacked = frame.stack(dropna=False)
    stacked.index = stacked.index.set_names(["date", "ticker"])
    return stacked


def cross_sectional_zscore(values: pd.Series, clip: float | None = 3.0) -> pd.Series:
    """Per-date z-score normalization that preserves two-level index exactly."""
    if not isinstance(values.index, pd.MultiIndex):
        raise ValueError("Factor values must use MultiIndex (date, ticker)")

    grouped = values.groupby(level="date")
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    z = (values - mean) / (std + 1e-12)

    if clip is not None:
        z = z.clip(-clip, clip)

    return z


def cross_sectional_rank(values: pd.Series) -> pd.Series:
    """Per-date percentile rank in [0,1], preserving index."""
    return values.groupby(level="date").rank(pct=True)


@dataclass(frozen=True)
class FactorOutput:
    name: str
    values: pd.Series


class BaseFactor(ABC):
    """Abstract factor class."""

    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_raw(self, market_data: pd.DataFrame) -> pd.Series:
        """Compute raw factor values from market data."""

    def compute(self, market_data: pd.DataFrame, normalize: bool = True) -> pd.Series:
        """Compute factor values and optionally apply cross-sectional z-score."""
        data = ensure_market_data(market_data)
        raw = self.compute_raw(data)

        if not isinstance(raw.index, pd.MultiIndex):
            raise ValueError(f"Factor {self.name} must return MultiIndex series")

        raw = raw.rename(self.name)
        values = cross_sectional_zscore(raw) if normalize else raw
        return values


class CompositeFactor(BaseFactor):
    """Weighted linear combination of base factors."""

    def __init__(self, factors: List[BaseFactor], weights: Iterable[float] | None = None, name: str = "Composite"):
        super().__init__(name=name)
        self.factors = factors
        w = np.array(list(weights) if weights is not None else [1 / len(factors)] * len(factors), dtype=float)
        if len(w) != len(factors):
            raise ValueError("weights length must match number of factors")
        self.weights = w / w.sum()

    def compute_raw(self, market_data: pd.DataFrame) -> pd.Series:
        series = []
        for factor, weight in zip(self.factors, self.weights):
            series.append(factor.compute(market_data, normalize=True) * weight)
        return pd.concat(series, axis=1).sum(axis=1)
