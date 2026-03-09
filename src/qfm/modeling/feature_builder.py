"""Feature construction for multi-factor research."""

from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from qfm.data.contracts import ensure_market_data
from qfm.factors.base import BaseFactor


def build_factor_matrix(
    market_data: pd.DataFrame,
    factors: Iterable[BaseFactor],
    normalize: bool = True,
) -> pd.DataFrame:
    """Build multi-factor matrix with index (date, ticker) and factor columns."""
    data = ensure_market_data(market_data)

    outputs: Dict[str, pd.Series] = {}
    for factor in factors:
        outputs[factor.name] = factor.compute(data, normalize=normalize)

    matrix = pd.DataFrame(outputs)
    matrix = matrix.sort_index()
    matrix.index = matrix.index.set_names(["date", "ticker"])
    return matrix


def latest_factor_snapshot(factor_matrix: pd.DataFrame) -> pd.Series:
    """Extract latest cross-sectional factor score by ticker."""
    if factor_matrix.empty:
        raise ValueError("factor_matrix is empty")
    latest_date = factor_matrix.index.get_level_values("date").max()
    latest = factor_matrix.loc[latest_date]
    return latest.mean(axis=1)
