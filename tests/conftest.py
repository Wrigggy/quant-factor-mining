"""Pytest fixtures for offline deterministic tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qfm.data.fetch import generate_synthetic_market_data  # noqa: E402


@pytest.fixture(scope="session")
def market_data() -> pd.DataFrame:
    return generate_synthetic_market_data(
        start="2020-01-01",
        end="2023-12-31",
        tickers=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"],
        seed=7,
    )
