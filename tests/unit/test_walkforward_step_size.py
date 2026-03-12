from __future__ import annotations

import pandas as pd
import pytest

from qfm.modeling.walkforward import generate_folds


def test_generate_folds_step_size_changes_fold_count():
    dates = pd.bdate_range("2024-01-01", periods=40)

    folds_default = generate_folds(dates, train_size=12, test_size=6)
    folds_step_1 = generate_folds(dates, train_size=12, test_size=6, step_size=1)
    folds_step_21 = generate_folds(dates, train_size=12, test_size=6, step_size=21)

    assert len(folds_step_1) > len(folds_default) > len(folds_step_21)
    assert folds_default[0].train_start == folds_step_1[0].train_start == folds_step_21[0].train_start


def test_generate_folds_rejects_non_positive_step_size():
    dates = pd.bdate_range("2024-01-01", periods=30)
    with pytest.raises(ValueError, match="step_size must be positive"):
        generate_folds(dates, train_size=10, test_size=5, step_size=0)
