from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qfm.backtest.engine import _target_weights_from_scores


def test_score_tilted_prefers_higher_scores_and_sums_to_one():
    row = pd.Series({"A": 2.5, "B": 1.2, "C": 0.4, "D": -0.5})
    universe = pd.Index(["A", "B", "C", "D"])

    w = _target_weights_from_scores(
        score_row=row,
        top_n=3,
        universe=universe,
        weighting_mode="score_tilted",
        score_temperature=0.8,
        max_single_weight=1.0,
    )

    assert np.isclose(float(w.sum()), 1.0)
    assert w["A"] > w["B"] > w["C"] > 0.0
    assert np.isclose(float(w["D"]), 0.0)


def test_score_tilted_respects_max_single_weight():
    row = pd.Series({"A": 8.0, "B": 0.8, "C": 0.2})
    universe = pd.Index(["A", "B", "C"])

    w = _target_weights_from_scores(
        score_row=row,
        top_n=3,
        universe=universe,
        weighting_mode="score_tilted",
        score_temperature=0.5,
        max_single_weight=0.45,
    )

    assert np.isclose(float(w.sum()), 1.0)
    assert float(w.max()) <= 0.45 + 1e-12
    assert float(w.min()) >= 0.0


def test_score_tilted_raises_for_infeasible_max_weight():
    row = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
    universe = pd.Index(["A", "B", "C"])

    with pytest.raises(ValueError, match="max_single_weight is too small"):
        _target_weights_from_scores(
            score_row=row,
            top_n=3,
            universe=universe,
            weighting_mode="score_tilted",
            score_temperature=1.0,
            max_single_weight=0.30,
        )
