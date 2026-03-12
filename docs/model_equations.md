# Factor Model Equations and Intuition

## Data and Labels

Input data is a daily OHLCV panel indexed by `(date, ticker)`.

- `open`, `high`, `low`, `close`, `volume`
- one row per asset per business day

Label is 1-day forward close return:

\[
y_{i,t} = \frac{C_{i,t+1}}{C_{i,t}} - 1
\]

where:
- \(i\): asset
- \(t\): date
- \(C_{i,t}\): close price

## Three Factors

### 1) Momentum (medium-term continuation)

\[
f^{mom}_{i,t} = \frac{C_{i,t-\text{skip}}}{C_{i,t-(\text{lookback}+\text{skip})}} - 1
\]

Interpretation: assets with stronger medium-term past returns tend to continue.

### 2) Mean Reversion (short-term reversal)

\[
f^{mr}_{i,t} = -\left(\frac{C_{i,t}}{C_{i,t-\text{lookback}}} - 1\right)
\]

Interpretation: recent short-term losers receive higher scores (potential rebound).

### 3) Low Volatility

\[
f^{lv}_{i,t} = -\sigma_{i,t}(\text{return}, \text{window})
\]

where \(\sigma\) is rolling return standard deviation (annualized in the implementation).

Interpretation: lower realized volatility gets higher score.

## Normalization, IC Weights, Composite Score

Each factor is normalized cross-sectionally per date:

\[
z^j_{i,t} = \frac{f^j_{i,t} - \mu^j_t}{\sigma^j_t + \epsilon}
\]

Then, in each train window, compute daily cross-sectional IC:

\[
IC_j(t) = \text{SpearmanCorr}_i\left(z^j_{i,t}, y_{i,t}\right)
\]

Train mean IC per factor:

\[
\overline{IC}_j = \frac{1}{T_{train}}\sum_t IC_j(t)
\]

Factor coefficients (train-only):

\[
w_j = \frac{\max(\overline{IC}_j, 0)}{\sum_k \max(\overline{IC}_k, 0)}
\]

If all train IC means are non-positive, the implementation falls back to equal factor weights.

Composite score per asset/day:

\[
S_{i,t} = \sum_j w_j z^j_{i,t}
\]

## Portfolio Construction and Execution Timing

At each rebalance date:
1. Rank assets by \(S_{i,t}\)
2. Select top `N`
3. Assign equal weights to selected assets

\[
x_{i,t}=
\begin{cases}
\frac{1}{N}, & i \in \text{TopN}(S_t) \\
0, & \text{otherwise}
\end{cases}
\]

Timing is leakage-safe:
- signal computed at date \(t\)
- rebalance at close of \(t\)
- new weights active from \(t+1\)

## Benchmark and Attribution

Benchmark mode:
- preferred: SPY daily return
- fallback: equal-weight benchmark from in-universe asset returns

Attribution metrics include:
- benchmark return
- excess return
- alpha
- beta
- tracking error
- information ratio

## Visual Equation Map

```text
Daily OHLCV panel
    -> factor raw values (momentum, mean-reversion, low-volatility)
    -> per-date cross-sectional z-score
    -> train-window IC per factor
    -> factor coefficients from mean IC
    -> composite score S(i,t)
    -> top-N ranking and equal-weight portfolio
    -> execute at close(t), active from t+1
    -> strategy returns vs benchmark returns
    -> attribution (alpha / beta / tracking error / IR)
```
