# Deep Research: How Top Quant Funds Mine and Discover Alpha Factors

*Research compiled April 2026 | For guiding the redesign of quant-factor-mining framework*

---

## 1. What Exactly Is "Factor Mining"?

Factor mining is **both** discovering new factor formulas **and** finding optimal combinations of known factors. The industry uses the term to cover the entire pipeline from raw data to deployed trading signal.

### The Two Layers

**Layer 1 — Alpha Expression Discovery:**
Discovering new mathematical formulas (expressions) that transform raw market data (price, volume, fundamentals, alternative data) into predictive signals. Example: `rank(decay_linear(correlation(vwap, sum(adv5, 26), 5), 7))` — a formula that computes a time-decayed correlation between VWAP and smoothed volume, then ranks stocks cross-sectionally.

**Layer 2 — Alpha Combination (Mega-Alpha):**
Combining hundreds to millions of individually weak alphas into a single composite signal ("mega-alpha") using dynamic weighting. WorldQuant explicitly frames this as mining "hundreds of thousands, millions and even billions of alphas and combining them into a unified mega-alpha."

### The Actual Workflow (Industry Standard)

```
Raw Data → Feature Engineering → Alpha Expression Generation → Single-Alpha Evaluation
    → Overfitting Controls → Alpha Library → Dynamic Combination → Portfolio Construction
    → Risk Management → Execution → Performance Monitoring → Feedback Loop
```

This is not a one-shot process. It is a **closed-loop** system where performance feedback guides the next round of hypothesis generation.

---

## 2. How Top Quant Funds Discover New Factors

### Renaissance Technologies
- **Approach**: Fully automated signal discovery. Scientists (physicists, mathematicians, astronomers — no MBAs) build systems that find patterns, not individual strategies.
- **Core techniques**: Hidden Markov Models (HMMs) using the Baum-Welch algorithm to detect regime changes; Bayesian inference for continuous probability updates; time-series statistical methods for non-linear relationship detection.
- **Key insight**: Right on only ~50.75% of trades, but across millions of trades daily (150,000–300,000 executions/day), that tiny edge compounds massively.
- **Philosophy**: Portfolio-level statistical arbitrage — long/short portfolios hedging out market risk, sector risk, and all statistically predictable risks. The entire operation is automated end-to-end.

### Two Sigma
- **Approach**: Technology-first. Employs data scientists over finance MBAs.
- **Key differentiator**: Master of alternative data — satellite imagery, credit card flows, geolocation, patent filings.
- **Signal discovery**: Crowdsourcing competitions to discover new trading signals; heavy ML infrastructure.
- **Philosophy**: Build engineering infrastructure that enables rapid hypothesis testing at scale.

### DE Shaw
- **Approach**: Combines structural analysis with quantitative research.
- **Focus**: Identifying structural market inefficiencies through rigorous statistical methods.
- **Integration**: Blends systematic and discretionary approaches, with quant research informing both.

### Citadel
- **Equity Quantitative Research (EQR)**: Applies "rigorous statistical methods to diverse datasets to identify novel and idiosyncratic signals grounded in economic practicality."
- **Global Quantitative Strategies (GQS)**: Algorithmic strategies across Equities, Futures, Fixed Income, and FX.
- **Philosophy**: Constant experimentation — "deploying novel techniques or reframing existing problems." Behavioral phenomena translated into trade opportunities across multiple time horizons.

### AQR Capital Management
- **Approach**: Factor investing grounded in academic research. Systematic, disciplined, rules-based.
- **Core factors**: Value, Momentum, Quality, Low Volatility — but with proprietary enhancements.
- **Philosophy**: "Fact, Fiction, and Factor Investing" — rigorous debunking of factor myths; evidence-based approach. Published extensive research showing factor premia persist but require discipline.

### Man Group (AHL)
- **Approach**: ML-driven since 2014 in live trading. Receives 1.5 billion data ticks/day.
- **NLP platform**: Built a firm-wide NLP research platform (2020) making text-based research as accessible as numerical pattern recognition.
- **Alternative data**: Weather forecasts for energy/crop markets, broker recommendations, news sentiment.
- **Key insight**: "Combining numerous varied weak information sources into investment systems with greater signalling power than any individual source."
- **Innovation**: Nearly tripled model count in 2 years (as of end-2023). Current alphas have only 40-50% correlation with pre-2018 alphas, indicating substantial methodology evolution.
- **Cross-disciplinary**: Applied astronomy ML methods to financial signal extraction (via Oxford-Man Institute collaboration).

### WorldQuant
- **Approach**: Crowdsourced alpha mining at massive scale. See Section 5 for deep dive.
- **Scale**: Millions to billions of formulaic alphas, combined into mega-alpha.
- **Platform**: BRAIN — open to external researchers worldwide.

### Summary: The Five Approaches to Factor Discovery

| Approach | Who Uses It | Pros | Cons |
|----------|-------------|------|------|
| **Genetic Programming / Symbolic Regression** | WorldQuant, academic researchers | Interpretable formulas, huge search space | Overfitting risk, slow convergence |
| **Deep Learning (latent factors)** | Two Sigma, Man AHL | Captures non-linear patterns | Black box, needs massive data |
| **Economic Intuition + Hypothesis Testing** | AQR, Citadel EQR | Grounded in theory, more robust | Slower, human-bottlenecked |
| **Alternative Data Mining** | Two Sigma, Man AHL, Renaissance | Unique signals, less crowded | Data cost, processing complexity |
| **RL / MCTS-based Search** | Academic frontier, some prop shops | Optimizes for portfolio-level synergy | Computationally expensive |

---

## 3. A Modern Factor Mining Pipeline

### Stage 1: Data Ingestion
**Sources:**
- **Price/Volume**: OHLCV, VWAP, returns, average daily volume (adv5, adv20)
- **Fundamentals**: Balance sheet, income statement, cash flow (quarterly)
- **Options**: Implied volatility surfaces, call/put breakevens
- **Alternative**: News sentiment, analyst recommendations, social media buzz, satellite imagery, credit card flows, weather, patent filings, geolocation
- **Scale**: WorldQuant BRAIN hosts 85,000+ data types

### Stage 2: Alpha Expression Generation

**Method A — Formulaic Alpha Mining (Symbolic):**
Generate expression trees combining operators and data fields.

Operator taxonomy:
- **Cross-sectional**: `rank`, `zscore`, `scale`, `sign`, `CSRank`
- **Time-series**: `ts_rank`, `ts_argmax`, `ts_argmin`, `ts_mean`, `ts_sum`, `ts_std`, `ts_max`, `ts_min`, `decay_linear`, `delta`, `delay`, `EMA`, `WMA`
- **Statistical**: `correlation`, `covariance`, `stddev`, `skew`, `kurtosis`
- **Arithmetic/Logic**: `abs`, `log`, `sign`, `signedpower`, `if_else`, `trade_when`
- **Group**: Operations within sector/industry groups
- **Neutralization**: `vector_neut` (orthogonal neutralization against risk factors)

Data fields: `open`, `high`, `low`, `close`, `volume`, `vwap`, `returns`, `adv[N]`, `cap`, fundamental fields

**Example formula** (from Kakushadze 101 Alphas):
```
Alpha#98 = rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088))
         - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15),
           20.8187), 8.62571), 6.95668), 8.07206))
```

**Method B — RL/MCTS-based Generation:**
- **AlphaGen** (KDD 2023): Formulates alpha mining as an MDP; uses Maskable PPO with LSTM to construct expressions token-by-token in Reverse Polish Notation. Reward = IC of the combined alpha set (not individual alpha), promoting synergy.
- **RiskMiner**: MCTS as a sampler providing training data for risk-seeking policy optimization.
- **Alpha2**: Deep RL discovering logical formulaic alphas with compositional structure.

**Method C — LLM-driven (2024-2025 frontier):**
- **AlphaAgent**: Three-agent system (Idea Agent → Factor Agent → Eval Agent). Idea Agent generates hypotheses via chain-of-thought; Factor Agent translates to ASTs; Eval Agent backtests.
- **Alpha-GPT**: Human-AI interactive alpha mining — researcher provides intuition, LLM outputs creative alpha expressions.
- **AlphaForge** (AAAI 2025): GAN-like generator produces alpha expression matrices via gumbel-softmax; predictor guides search.

**Method D — Deep Learning (Latent Factors):**
- Not formulaic — learns implicit factors from raw data (see Section 7).

### Stage 3: Single-Alpha Evaluation

**Core Metrics:**
| Metric | Formula | Good Threshold | What It Measures |
|--------|---------|---------------|------------------|
| **IC** | `Pearson(factor_values, forward_returns)` | > 0.03–0.05 | Predictive power |
| **Rank IC** | `Spearman(factor_ranks, return_ranks)` | > 0.03–0.05 | Rank predictive power |
| **ICIR** | `mean(IC) / std(IC)` | > 0.1 | Signal stability |
| **Rank ICIR** | `mean(RankIC) / std(RankIC)` | > 0.1 | Rank signal stability |
| **Sharpe Ratio** | `mean(returns) / std(returns) * sqrt(252)` | > 1.6 (WQ BRAIN) | Risk-adjusted returns |
| **Turnover** | `% portfolio traded daily` | 5–15% | Tradability |
| **Fitness** | Composite of above | > 0.85 (WQ BRAIN) | Overall quality |

**AlphaForge thresholds for factor pool admission:** IC > 3%, ICIR > 0.1

### Stage 4: Overfitting Controls (See Section 6 for details)
- Deflated Sharpe Ratio
- Combinatorial Purged Cross-Validation
- Multiple testing corrections
- Out-of-sample holdout
- Alpha originality scoring (AST similarity)

### Stage 5: Alpha Combination → Mega-Alpha

**AlphaForge dynamic combination (daily):**
1. Filter: Keep factors with IC > threshold on recent window
2. Rank by recent IC performance
3. OLS regression: Fit top-N factors to predict returns
4. Generate weighted composite signal (dynamic weights updated daily)

**AlphaGen synergistic approach:**
- RL reward = IC of the linearly combined alpha set
- Optimizes for portfolio-level performance, not individual alpha quality
- Naturally produces diverse, complementary factors

### Stage 6: Portfolio Construction & Risk Management
- Mean-variance optimization with factor-model risk
- Neutralize against known risk factors (market, sector, size, etc.)
- Position limits, turnover constraints, capacity constraints
- Transfer coefficient: Typically 0.3–0.8 in practice (constraints reduce signal transfer)

---

## 4. Alpha Factors vs. Risk Factors

### Definitions

**Alpha Factors (Trading Signals):**
- Predictive signals you **trade on** to generate excess returns
- Short-lived, proprietary, idiosyncratic
- Examples: momentum variants, mean-reversion signals, sentiment signals, alternative data signals
- Evaluated by IC, Sharpe, turnover, capacity
- Goal: Maximize information ratio

**Risk Factors (Barra-style):**
- Systematic exposures you **hedge against** or use for attribution
- Long-lived, well-known, published
- Examples: Market (MKT), Size (SMB), Value (HML), Momentum (UMD), Quality, Low Volatility
- Used in risk models to decompose portfolio variance
- Goal: Explain and control risk

### The Critical Distinction in Practice

> "Risk factors represent market risk exposures that clients don't want, whereas alpha represents uncorrelated excess returns."

**The lifecycle problem:** Today's alpha becomes tomorrow's risk factor. Many original quant "alpha factors" (value, momentum, quality) are now widely recognized as factor premia — available cheaply via ETFs. True alpha is what **cannot** be explained by known risk factors.

**Practical test:** Run a regression of your signal's returns against Fama-French + Momentum + Quality + Low-Vol factors. If the intercept (alpha) is statistically significant, you have a genuine alpha factor. If returns are fully explained by factor loadings, you have repackaged risk premia.

**Portable alpha strategy:** Separate alpha generation from beta exposure. Hedge out all known factor exposures; the residual is your tradeable alpha.

### How Quant Funds Handle This

1. **Build alpha signals** (proprietary, short-horizon, idiosyncratic)
2. **Use risk models** (Barra, Axioma, or proprietary) to decompose portfolio risk
3. **Neutralize** alpha portfolios against all known risk factors
4. **Monitor** for alpha decay — when your alpha becomes a known factor, it's dying
5. **Attribute** P&L to alpha vs. factor exposure to ensure you're generating genuine skill

---

## 5. WorldQuant's Approach: Deep Dive

### Scale
WorldQuant mines "hundreds of thousands, millions and even billions of alphas" and combines them into a unified mega-alpha. This is industrial-scale signal mining.

### The BRAIN Platform

**Infrastructure:**
- 85,000+ data types
- Hundreds of operator functions
- Custom backtesting simulator
- Open to external researchers worldwide

**Fast Expression Language:**
A proprietary pseudo-code language with two elements: **Data Fields** and **Operators**.

**Operator Categories:**
| Category | Examples |
|----------|---------|
| Cross-sectional | `rank`, `zscore`, `scale`, `sign` |
| Time-series | `ts_delta`, `ts_delay`, `ts_sum`, `ts_mean`, `ts_rank`, `ts_zscore`, `ts_stddev`, `ts_regression` |
| Group | Operations within sector/industry groupings |
| Conditional | `trade_when`, `if_else` |
| Neutralization | `vector_neut` (orthogonalize against risk factors) |
| Decay | `humpdecay` (volatility-based turnover reduction) |

**Data Categories:**
- Price/Volume: OHLC, returns, volume, VWAP
- Fundamentals: Balance sheet, income statement, cash flow
- Options: Implied volatility surfaces, call/put breakevens
- Alternative: News sentiment, analyst recommendations, social media

### Evaluation Criteria (BRAIN)

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 1.6 (delay-0), > 2.0 strong |
| Turnover | 5–15% daily |
| Fitness | > 0.85 |
| Drawdown | Minimize |
| Information Ratio | `IR = Sharpe / sqrt(252)` |

**Relationship:** `Sharpe = sqrt(252) * IR`

### The 101 Formulaic Alphas (Kakushadze, 2015)

The foundational public reference for formulaic alpha mining:
- 101 real-world alpha formulas published with WorldQuant's permission
- Average holding period: 0.6–6.4 days
- Average pairwise correlation: 15.9% (low — diverse set)
- Returns correlated with volatility, not turnover
- Mostly price-volume based, some use fundamental data
- Nesting depth up to 6 levels

### Combination Strategy

1. Submit many diverse alphas (breadth over depth)
2. Maintain low pairwise correlation between alphas
3. Layer signals: "Price-volume + fundamental is best"
4. Use neutralization (`vector_neut`) to orthogonalize against known factors
5. Dynamic weighting based on recent performance
6. Truncation value = 0.01 for position sizing diversity

---

## 6. The Multiple Testing / Overfitting Problem

This is the single most important unsolved challenge in factor mining. As Lopez de Prado states: "Most discoveries in empirical finance are false, as a consequence of selection bias under multiple testing."

### The Core Problem

If you test N factor formulas and select the best one, the expected Sharpe ratio of the best is:

```
E[max(SR_1, ..., SR_N)] ≈ sqrt(V[SR]) * ((1-γ) * Φ^(-1)[1-1/N] + γ * Φ^(-1)[1-1/(N*e)])
```

Where γ ≈ 0.5772 (Euler-Mascheroni constant). Even with zero-skill strategies, testing enough hypotheses **guarantees** finding apparently significant results.

**Lopez de Prado's key finding:** "As few as 3 trials are enough to produce an investment strategy that is likely false" for strategies with Sharpe ratios of 1.5 over 10-year backtests.

### Solution 1: Deflated Sharpe Ratio (DSR)

The DSR corrects for selection bias and non-normal returns:

```
DSR = Φ( (SR* - SR₀) * sqrt(T-1) / sqrt(1 - γ₃*SR₀ + (γ₄-1)/4 * SR₀²) )
```

Where:
- `SR*` = observed Sharpe ratio
- `SR₀` = expected maximum Sharpe from N unskilled strategies
- `γ₃` = return skewness
- `γ₄` = return kurtosis
- `T` = sample length

**Key difference from Bonferroni/Sidak:** DSR accounts for variance of Sharpe estimates, number of trials, and their effective independence (estimated via clustering), not just the count of tests.

### Solution 2: Combinatorial Purged Cross-Validation (CPCV)

Standard k-fold CV is invalid for financial time series (violates IID assumption). CPCV addresses this:

1. **Purging**: Remove training samples that overlap with the test set's label horizon
2. **Embargo**: Exclude a buffer of observations at test-set boundaries to prevent information leakage
3. **Combinatorial**: Systematically construct all possible train/test splits from N groups, choosing k for testing

**Result:** Lower Probability of Backtest Overfitting (PBO) and superior DSR test statistics compared to walk-forward or naive CV.

### Solution 3: Alpha Originality Scoring (AlphaAgent)

Structural overfitting control via AST matching:

```
s(fi, fj) = max{|ti| : ti ⊆ T(fi), tj ⊆ T(fj), ti ≅ tj}
```

Measures the largest common subtree between two factor ASTs. High similarity to existing alphas → penalized.

### Solution 4: Regularized Exploration (AlphaAgent)

Combined regularization:
```
R(f,h) = α₁*SL(f) + α₂*PC(f) + α₃*ER(f,h)
```
- `SL(f)` = symbolic length (complexity penalty)
- `PC(f)` = parameter count (free parameter penalty)
- `ER(f,h)` = novelty + hypothesis alignment

### Practical Guidelines

1. **Record all trials** — never test without logging
2. **Estimate effective N** via clustering (not raw trial count)
3. **Apply DSR** to every candidate strategy
4. **Use CPCV** instead of walk-forward for backtest validation
5. **Penalize complexity** — simpler alphas are less likely overfit
6. **Require economic intuition** — pure data-mining alphas are more likely false
7. **Out-of-sample holdout** — always reserve unseen data
8. **Paper trading period** — live verification before capital deployment

---

## 7. Modern ML Approaches to Factor Mining

### Autoencoders / Deep Latent Factor Models

**FactorVAE** (AAAI 2022):
- Integrates Dynamic Factor Model (DFM) with Variational Autoencoder (VAE)
- Prior-posterior learning: approximates optimal posterior factor model using future information during training
- Estimates both expected returns AND variances from the latent space
- 35% improvement in prediction accuracy vs. traditional dynamic models
- Key insight: The VAE's latent space naturally learns factors that explain cross-sectional return variation

**HireVAE** (2023):
- First deep learning online/adaptive factor model
- Hierarchical latent space embedding the relationship between market regime and stock-wise latent factors
- Two levels: market situation → stock-specific factors
- Adapts factors to regime changes without retraining

**RVRAE** (2024):
- Variational Recurrent Autoencoder for dynamic factor estimation
- Captures temporal dependencies in factor loadings

### Transformer-Based Models

**Quantformer** (2024):
- Attention mechanism applied to cross-sectional stock selection
- Self-attention captures complex inter-stock dependencies
- Result: 16.41% annualized excess return, 0.87 Sharpe, 15.02% max drawdown

**AlphaFormer**:
- End-to-end symbolic regression of alpha factors using Transformers
- Generates interpretable alpha expressions (not black-box predictions)
- Bridges the gap between deep learning power and formulaic interpretability

**Key finding from literature review (187 studies, 2020-2024):** Models with temporal dependency (FactorVAE, ALSTM, Transformers) consistently outperform static ML models and traditional linear factor models.

### Reinforcement Learning for Portfolio Construction

**Multi-Agent RL:**
- Graph attention-based heterogeneous multi-agent RL framework
- Three specialized agents: risk assessment, return prediction, market environment perception
- Graph attention networks model time-varying asset correlations
- Result: 16.8% annualized returns, 1.34 Sharpe, 8.2% max drawdown

**AlphaGen (RL for Alpha Set Generation):**
- Maskable PPO with LSTM generates sets of synergistic formulaic alphas
- Reward = IC of the combined set, not individual alphas
- Naturally produces diverse, complementary factors

**Deep Q-Network for Portfolio Optimization:**
- DQN learns optimal rebalancing policies
- State space includes factor values, portfolio positions, market conditions

### Graph Neural Networks for Cross-Asset Signals

**Dynamic Wavelet Coherence GNN (WCG-RL, 2024):**
- Multi-graph representation based on wavelet coherence
- Captures dynamic time-frequency correlations between assets
- Integrated with deep RL for portfolio management

**GNN Asset Pricing Model:**
- Graph attention learns dynamic equity market network structure
- Recurrent CNN diffuses firm information through learned networks
- Long-short portfolio: 17% annualized return

**Key insight:** GNNs model how assets influence each other beyond simple correlation — e.g., if a tech stock rallies due to AI advances, GNNs predict ripple effects across semiconductor stocks and tech ETFs.

### The Emerging Stack (2024-2025)

```
LLM (hypothesis generation)
    → Symbolic Regression / RL (expression construction)
        → Deep Learning (non-linear enhancement)
            → GNN (cross-asset dependencies)
                → RL (portfolio-level optimization)
```

---

## 8. What Metrics Matter Most for Factor Quality

### The Complete Evaluation Framework

#### Primary Metrics

**1. Information Coefficient (IC)**
- `IC = Pearson(factor_values_t, returns_t+1)`
- Cross-sectional correlation between factor scores and forward returns
- Realistic range: 0.02–0.10 (anything > 0.15 is suspicious)
- IC of 0.05 is considered strong

**2. Rank IC (Spearman IC)**
- `RankIC = Spearman(factor_ranks_t, return_ranks_t+1)`
- More robust to outliers than Pearson IC
- Often preferred in practice

**3. ICIR (IC Information Ratio)**
- `ICIR = mean(IC) / std(IC)`
- Measures stability/consistency of the signal
- ICIR > 0.5 is generally strong

**4. IC Decay Profile**
- Plot IC at lag 1, 2, 3, ..., N days
- Reveals the factor's information horizon
- Fast decay → high turnover strategy (need low transaction costs)
- Slow decay → lower turnover, more capacity
- **Factor half-life**: Time for IC to drop to 50% of its peak value
- Optimal rebalancing period depends on the decay profile

#### Tradability Metrics

**5. Turnover**
- `Turnover = sum(|w_t - w_{t-1}|) / 2` (daily)
- Higher turnover = more transaction costs
- Must evaluate: `Net IC = IC - cost_per_unit_turnover * turnover`
- WorldQuant target: 5–15% daily

**6. Capacity**
- How much capital can trade the signal before impact erodes returns
- Function of: liquidity of traded instruments, turnover, uniqueness
- High-frequency signals: low capacity ($10M–$100M)
- Slow fundamental signals: high capacity ($1B+)
- Scaling law: `CPS ~ 1/turnover` (cents-per-share scales inversely with turnover)

#### Robustness Metrics

**7. Crowding**
- Is everyone trading this signal?
- Measured on relative basis (absolute capital in a strategy is hard to estimate)
- Factors with high barriers to entry maintain performance longer
- Crowded factors are prone to violent unwinds in tail events
- MSCI Integrated Factor Crowding Model: monitors factor crowding in real time

**8. Factor Orthogonality**
- Is the signal truly new, or repackaged momentum/value/etc.?
- Test: Regress factor returns on known factors (Fama-French + Momentum + Quality + Low-Vol)
- Significant intercept = genuinely new information
- AlphaForge target: return correlation with existing factors < threshold (CORR')
- Kakushadze 101 alphas: average pairwise correlation = 15.9%

**9. Stability Across Regimes**
- Test across bull/bear markets, high/low volatility, different macro environments
- HireVAE explicitly models regime-dependent factors

**10. Drawdown Profile**
- Maximum drawdown and drawdown duration
- A high-Sharpe factor with catastrophic drawdowns is still dangerous

#### The Fundamental Law of Active Management

```
IR = IC * sqrt(BR) * TC
```

Where:
- `IR` = Information Ratio (target metric)
- `IC` = Information Coefficient (forecast skill)
- `BR` = Breadth (number of independent bets per year)
- `TC` = Transfer Coefficient (signal-to-portfolio efficiency, typically 0.3–0.8)

**Implication:** You can either have very high IC (hard) or very high breadth (many independent bets). WorldQuant's strategy is extreme breadth — millions of weak alphas combined.

---

## Key References

### Papers
- Kakushadze, Z. (2016). *101 Formulaic Alphas*. [arXiv:1601.00991](https://arxiv.org/pdf/1601.00991)
- Lopez de Prado, M. (2018). *The Deflated Sharpe Ratio*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- Lopez de Prado, M. (2018). *A Data Science Solution to the Multiple-Testing Crisis in Financial Research*. [SSRN/AQR](https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/JFDS_Winter2019_A-Data-Science-Solution-to-Multiple-Testing-Crisis---Lopez_de_Prado.pdf?sc_lang=en)
- Yu et al. (2023). *AlphaGen: Generating Synergistic Formulaic Alpha Collections via RL*. [KDD 2023 / GitHub](https://github.com/RL-MLDM/alphagen)
- Chen et al. (2025). *AlphaForge: Mine and Dynamically Combine Formulaic Alpha Factors*. [AAAI 2025](https://arxiv.org/html/2406.18394v1)
- Huang et al. (2025). *AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration*. [arXiv](https://arxiv.org/html/2502.16789v2)
- Li et al. (2024). *Alpha Mining via Warm Start Genetic Programming*. [arXiv](https://arxiv.org/html/2412.00896v1)
- Duan et al. (2022). *FactorVAE: Probabilistic Dynamic Factor Model Based on VAE*. [AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20369)
- Shi et al. (2023). *HireVAE: Hierarchical and Regime-Switch VAE Factor Model*. [arXiv](https://arxiv.org/abs/2306.02848)
- AQR (2023). *Fact, Fiction, and Factor Investing*. [AQR](https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/AQRJPMQuant23FactFictionandFactorInvesting.pdf?sc_lang=en)

### Industry Sources
- [WorldQuant BRAIN Platform](https://platform.worldquantbrain.com/)
- [Man Group: The Quant Renaissance](https://www.man.com/insights/the-quant-renaissance)
- [Man Group: The Rise of Machine Learning at Man AHL](https://www.man.com/insights/the-rise-of-machine-learning)
- [Citadel EQR](https://www.citadel.com/what-we-do/equities/equity-quantitative-research-eqr/)
- [Renaissance Technologies (Wikipedia)](https://en.wikipedia.org/wiki/Renaissance_Technologies)
- [CPCV Explained](https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method)
- [AlphaCFG: Grammar-Guided Alpha Discovery](https://arxiv.org/html/2601.22119)
- [RiskMiner: MCTS for Alpha Mining](https://arxiv.org/html/2402.07080v2)

---

## Implications for quant-factor-mining Framework Redesign

Based on this research, the framework should support:

1. **Formulaic alpha expression language** — A DSL with the standard operator taxonomy (cross-sectional, time-series, statistical, conditional, group, neutralization operators) operating on configurable data fields.

2. **Multiple discovery methods** — Not just one approach, but pluggable generators:
   - Genetic programming / symbolic regression (baseline)
   - RL-based (AlphaGen-style, optimizing for portfolio-level IC)
   - LLM-assisted (hypothesis → expression → evaluation loop)
   - Manual hypothesis entry (economic intuition mode)

3. **Rigorous evaluation pipeline** — IC, RankIC, ICIR, RankICIR, turnover, capacity estimate, decay profile, factor orthogonality test, regime stability.

4. **Overfitting controls as first-class citizens** — DSR, CPCV, alpha originality scoring, complexity penalties. These are not optional add-ons.

5. **Dynamic combination** — Not static factor weights. Daily re-evaluation, OLS or ridge regression combination, with diversity constraints.

6. **Risk factor neutralization** — Built-in support for neutralizing against standard risk factors before evaluating alpha contribution.

7. **Alpha library / zoo management** — Track all discovered factors, their lineage, performance over time, pairwise correlations, and decay status.
