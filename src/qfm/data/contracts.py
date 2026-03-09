"""Data contracts and schema validation for market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class DataContractError(ValueError):
    """Raised when data does not satisfy the project data contract."""


@dataclass(frozen=True)
class DataContract:
    """Canonical data contract for market data frames.

    - Index: MultiIndex with names (date, ticker)
    - Columns: at least OHLCV
    - Sorted and unique index
    """

    required_columns: Sequence[str] = REQUIRED_OHLCV_COLUMNS

    def validate(self, data: pd.DataFrame, allow_empty: bool = False) -> None:
        """Validate input data frame against the market data contract."""
        if not isinstance(data, pd.DataFrame):
            raise DataContractError("Market data must be a pandas DataFrame")

        if data.empty and not allow_empty:
            raise DataContractError("Market data is empty")

        if not isinstance(data.index, pd.MultiIndex):
            raise DataContractError("Market data index must be MultiIndex (date, ticker)")

        if data.index.nlevels != 2:
            raise DataContractError("Market data MultiIndex must have exactly 2 levels")

        idx_names = tuple(data.index.names)
        if idx_names != ("date", "ticker"):
            raise DataContractError(
                f"Market data index names must be ('date', 'ticker'), got {idx_names}"
            )

        missing = [col for col in self.required_columns if col not in data.columns]
        if missing:
            raise DataContractError(f"Market data missing required columns: {missing}")

        if not data.index.is_unique:
            raise DataContractError("Market data index contains duplicates")

        if not data.index.is_monotonic_increasing:
            raise DataContractError("Market data index must be sorted by (date, ticker)")

        if "high" in data.columns and "low" in data.columns:
            invalid = (data["high"] < data["low"]).sum()
            if invalid > 0:
                raise DataContractError(f"Found {invalid} rows where high < low")


DEFAULT_CONTRACT = DataContract()


def ensure_market_data(data: pd.DataFrame, allow_empty: bool = False) -> pd.DataFrame:
    """Return a normalized view of market data after validating contract."""
    normalized = data.copy()

    if not isinstance(normalized.index, pd.MultiIndex):
        raise DataContractError("Market data index must be MultiIndex (date, ticker)")

    # Normalize index dtypes and ordering for deterministic downstream behavior.
    level_date = pd.to_datetime(normalized.index.get_level_values("date"))
    level_ticker = normalized.index.get_level_values("ticker").astype(str)
    normalized.index = pd.MultiIndex.from_arrays(
        [level_date, level_ticker],
        names=["date", "ticker"],
    )
    normalized = normalized.sort_index()

    DEFAULT_CONTRACT.validate(normalized, allow_empty=allow_empty)
    return normalized


def ensure_wide_prices(close_wide: pd.DataFrame) -> pd.DataFrame:
    """Validate wide price frame (date index × ticker columns)."""
    if not isinstance(close_wide, pd.DataFrame):
        raise DataContractError("Wide prices must be a pandas DataFrame")
    if close_wide.empty:
        raise DataContractError("Wide prices are empty")
    if not isinstance(close_wide.index, pd.DatetimeIndex):
        raise DataContractError("Wide prices index must be DatetimeIndex")
    if close_wide.columns.empty:
        raise DataContractError("Wide prices must include at least one ticker column")

    close_wide = close_wide.sort_index()
    close_wide.columns = close_wide.columns.astype(str)
    return close_wide


def check_columns(data: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise if any required column is missing from data."""
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise DataContractError(f"Missing required columns: {missing}")
