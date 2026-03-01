# Pending Improvements — quant-factor-mining

Generated: 2026-02-28
Status: Issue #1 (memory leaks) resolved. All others pending.

---

## RESOLVED

### Issue #1 — Memory Leaks in Adaptive System
**Files:** `src/factors/adaptive_composite.py`
**Fix applied:**
- `weight_history` now capped at `ic_window * 2` entries (line ~114)
- `regime_history` now capped at `ic_window * 2` entries (line ~575)
- `ic_history` and `factor_buffer`/`price_buffer` were already pruned

---

## TIER 1 — Critical (Fix Before Live Trading)

### Issue #2 — Silent Failures in IC Calculation
**File:** `src/factors/adaptive_composite.py:397-402`
**Problem:** When aligned data < 10 stocks, IC update is skipped with only a warning.
Weights silently go stale — could persist for days undetected.
**Fix:**
- Track consecutive IC skip count per factor
- If skips exceed `forward_period` consecutive days, emit a `logger.error`
- Optionally expose a `stale_factors` property for monitoring

```python
# In AdaptiveCompositeManager.__init__:
self._ic_skip_count: Dict[str, int] = {}

# In compute_factor_with_update, after the continue:
self._ic_skip_count[factor_name] = self._ic_skip_count.get(factor_name, 0) + 1
if self._ic_skip_count[factor_name] >= self.forward_period:
    logger.error(
        f"Factor {factor_name} IC has not updated for "
        f"{self._ic_skip_count[factor_name]} consecutive periods — weights are stale!"
    )
# On successful IC update, reset:
self._ic_skip_count[factor_name] = 0
```
**Effort:** ~1 hour

---

### Issue #3 — Bare Exception Catching
**File:** `src/evaluation/alphalens_wrapper.py:421`
**Problem:** `except:` catches everything including `KeyboardInterrupt`, `SystemExit`.
Makes debugging impossible and can mask critical errors.
**Fix:**
```python
# Before:
except:
    pass

# After:
except Exception as e:
    logger.warning(f"alphalens operation failed: {e}")
```
**Effort:** 30 minutes — search for all bare `except:` in the codebase and replace.

---

### Issue #4 — Missing Covariance Matrix Validation
**File:** `src/optimization/risk_models.py`
**Problem:** No check that covariance matrix is positive semi-definite before passing
to optimizer. Can cause silent numerical failures in portfolio optimization.
**Fix:**
```python
import numpy as np

def validate_covariance(cov_matrix: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    if eigenvalues.min() < -1e-8:
        # Clip negative eigenvalues (nearest PSD projection)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 0)
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        logger.warning("Covariance matrix was not PSD — applied nearest PSD projection")
    return cov_matrix
```
**Effort:** ~1 hour

---

## TIER 2 — Important (Fix Before Scaling)

### Issue #5 — Inefficient Loops in Volatility Factor
**File:** `src/factors/volatility.py:222-242`
**Problem:** Sequential per-ticker loop for rolling covariance.
On 100+ stock universes this is 10-100x slower than vectorized.
**Fix:** Replace ticker loop with vectorized pandas rolling operations:
```python
# Before (slow):
for ticker in returns.columns:
    cov = ticker_returns.rolling(self.window).cov(aligned_market)

# After (fast):
# Stack market returns to align with returns DataFrame
market_aligned = pd.DataFrame(
    {col: aligned_market for col in returns.columns},
    index=returns.index
)
rolling_cov = returns.rolling(self.window).cov(market_aligned).iloc[::len(returns.columns)]
```
**Effort:** 3-4 hours (requires careful testing to verify output equivalence)

---

### Issue #6 — Missing Input Validation
**File:** `src/factors/base.py:33-47` and all factor `calculate()` methods
**Problem:** No check for required columns (`close`, `volume`, etc.) before unstack.
Results in cryptic `KeyError` instead of clear validation messages.
**Fix:** Add a validation helper in `BaseFactor`:
```python
REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

def _validate_data(self, data: pd.DataFrame, required: list = None):
    cols = required or ['close']
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"{self.name}: missing required columns {missing}")
    if not isinstance(data.index, pd.MultiIndex):
        raise ValueError(f"{self.name}: data must have MultiIndex (date, ticker)")
```
Call `self._validate_data(data)` at the top of each `calculate()`.
**Effort:** 2-3 hours

---

### Issue #7 — Code Duplication in Factor Implementations (~30%)
**Files:** `src/factors/momentum.py`, `src/factors/mean_reversion.py`, `src/factors/volatility.py`
**Problem:** All factors repeat:
- `unstack()` → calculate → `stack()` pattern
- Annualization logic (`* np.sqrt(252)`)
- Negative sign convention for short signals
**Fix:** Add utility methods to `BaseFactor`:
```python
def _unstack_close(self, data: pd.DataFrame) -> pd.DataFrame:
    return data['close'].unstack(level='ticker')

def _stack_result(self, df: pd.DataFrame) -> pd.Series:
    return df.stack().rename(self.name)

def _annualize_vol(self, daily_vol: pd.Series, periods: int = 252) -> pd.Series:
    return daily_vol * np.sqrt(periods)
```
**Effort:** 4-5 hours (refactor + regression test all factors)

---

### Issue #8 — No Monitoring / Alerting System
**Problem:** No way to detect degraded factor performance, data feed issues,
or weight staleness in production without manual inspection.
**Fix:** Add a lightweight `FactorMonitor` class:
- Track rolling IC per factor, alert if ICIR drops below threshold
- Alert on consecutive IC skip (see Issue #2)
- Log weight concentration (Herfindahl index) to detect degenerate solutions
- Optionally write metrics to SQLite for persistence
**Effort:** 4-6 hours

---

## TIER 3 — Nice to Have (Future Work)

### Issue #9 — Parameter Optimization Framework
**Problem:** No systematic way to tune `ic_window`, `decay_halflife`, `min_weight`, etc.
**Approach options:**
1. Grid search with walk-forward validation (simple, interpretable)
2. Bayesian optimization with `optuna` (faster convergence)
3. Genetic algorithm (good for large search spaces)
**Recommendation:** Start with `optuna` — minimal code, good defaults.
**Effort:** 20+ hours

---

### Issue #10 — ML Factor Combination
**Problem:** IC-weighted combination is linear. Non-linear combinations
(XGBoost/LightGBM on factor values) may capture interaction effects.
**Approach:** Train a gradient boosting model on factor values → forward returns,
use SHAP values as dynamic weights.
**Caveat:** Requires careful walk-forward validation to avoid overfitting.
**Effort:** 30+ hours

---

### Issue #11 — Remaining Jupyter Notebooks
**Status:** Only 1 of 6 planned notebooks complete.
**Missing:**
- `02_factor_analysis.ipynb` — IC/ICIR analysis across factors
- `03_regime_analysis.ipynb` — regime detection visualization
- `04_portfolio_construction.ipynb` — mean-variance optimization walkthrough
- `05_backtest_results.ipynb` — full backtest with tearsheet
- `06_live_trading_setup.ipynb` — production deployment guide
**Effort:** 10-15 hours

---

### Issue #12 — Real-Time Monitoring Dashboard
**Approach:** Streamlit app showing live factor weights, IC history, regime state.
**Effort:** 20+ hours

---

## Hardcoded Magic Numbers to Parameterize

| Location | Value | Suggested Config Key |
|---|---|---|
| `adaptive_composite.py:186` | `temperature = 2.0` | `softmax_temperature` |
| `volatility.py:85` | `min_periods=self.window // 2` | `min_periods_ratio` |
| `adaptive_composite.py:488` | `autocorr > 0.15` | `regime_autocorr_threshold` |
| `adaptive_composite.py:485` | `vol_percentile > 0.8` | `regime_vol_percentile` |

---

## Summary

| Tier | Issues | Est. Effort | Priority |
|---|---|---|---|
| Resolved | #1 memory leaks | Done | — |
| Tier 1 (critical) | #2, #3, #4 | ~2.5 hours | Before live trading |
| Tier 2 (important) | #5, #6, #7, #8 | ~15 hours | Before scaling |
| Tier 3 (nice to have) | #9–#12 | 80+ hours | Future |
