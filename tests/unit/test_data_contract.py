from __future__ import annotations

import pandas as pd
import pytest

from qfm.data.contracts import DataContractError, ensure_market_data


def test_market_data_contract_passes(market_data):
    normalized = ensure_market_data(market_data)
    assert isinstance(normalized.index, pd.MultiIndex)
    assert normalized.index.names == ["date", "ticker"]
    assert normalized.index.is_monotonic_increasing


def test_market_data_contract_rejects_empty():
    empty = pd.DataFrame()
    with pytest.raises(DataContractError):
        ensure_market_data(empty)
