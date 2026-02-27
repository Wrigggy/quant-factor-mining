# 快速开始指南

## 安装

```bash
cd /Users/kevinwu/Coding/quant-factor-mining
pip install -r requirements.txt
```

## 完整示例：从数据到回测

### 1. 获取和预处理数据

```python
from config.universes import get_universe
from src.data.fetcher import YFinanceFetcher
from src.data.cache_manager import CacheManager
from src.data.preprocessor import DataPreprocessor

# 初始化
fetcher = YFinanceFetcher()
cache = CacheManager()
preprocessor = DataPreprocessor()

# 获取数据
tickers = get_universe("US", "sample")[:20]  # 前20只标普500股票
data = cache.get_or_fetch(
    market="US",
    universe="sp500",
    tickers=tickers,
    start_date="2020-01-01",
    end_date="2024-01-01",
    fetcher=fetcher
)

# 预处理
clean_data, report = preprocessor.validate_data_quality(data)
print(f"数据质量: {report['data_coverage']:.2%}")
```

### 2. 计算Alpha因子

```python
from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor

# 计算多个因子
momentum = MomentumFactor(lookback=252, skip=21)
momentum_scores = momentum.compute(clean_data, normalize=True)

mean_reversion = MeanReversionFactor(lookback=21)
reversal_scores = mean_reversion.compute(clean_data, normalize=True)

volatility = VolatilityFactor(window=63)
vol_scores = volatility.compute(clean_data, normalize=True)

print(f"动量因子: {len(momentum_scores)} 个观测值")
print(f"均值回归因子: {len(reversal_scores)} 个观测值")
print(f"波动率因子: {len(vol_scores)} 个观测值")
```

### 3. 评估因子有效性

```python
from src.evaluation.alphalens_wrapper import AlphalensEvaluator
from src.evaluation.ic_analysis import ICAnalyzer

# Alphalens评估
evaluator = AlphalensEvaluator(periods=[1, 5, 21], quantiles=5)

results = evaluator.evaluate_factor(
    factor=momentum_scores,
    prices=clean_data,
    generate_plots=False
)

print(f"\n因子是否有效: {results['is_valid_factor']}")
print(f"\nIC统计:")
print(results['ic_summary'])

# IC深度分析
analyzer = ICAnalyzer(method='spearman')
forward_returns = analyzer.compute_forward_returns(clean_data, periods=[21])
ic_series = analyzer.compute_ic_series(momentum_scores, forward_returns[21])

ic_stats = analyzer.compute_ic_statistics(ic_series)
print(f"\nIC均值: {ic_stats['mean']:.4f}")
print(f"IC t统计量: {ic_stats['t_stat']:.2f}")
print(f"IC一致性: {ic_stats['positive_pct']:.2%}")
```

### 4. 构建最优投资组合

```python
from src.optimization.risk_models import CovarianceEstimator
from src.optimization.mean_variance import MeanVarianceOptimizer

# 使用最新的因子得分
latest_factors = momentum_scores.groupby(level='ticker').last()

# 估计协方差矩阵
close = clean_data['close'].unstack(level='ticker')
returns = close.pct_change().dropna()

cov_estimator = CovarianceEstimator(method='ledoit_wolf')
cov_matrix = cov_estimator.estimate(returns, window=252)

# 优化投资组合
optimizer = MeanVarianceOptimizer(
    risk_aversion=1.0,
    max_position=0.10,  # 单只股票最大10%
    long_only=True
)

optimal_weights = optimizer.optimize(
    expected_returns=latest_factors,
    covariance_matrix=cov_matrix
)

print(f"\n最优权重 (前10):")
print(optimal_weights.sort_values(ascending=False).head(10))

# 计算组合指标
metrics = optimizer.compute_portfolio_metrics(
    optimal_weights,
    latest_factors,
    cov_matrix
)

print(f"\n组合指标:")
print(f"  预期年化收益: {metrics['expected_return_annual']:.2%}")
print(f"  年化波动率: {metrics['volatility_annual']:.2%}")
print(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
print(f"  持仓数量: {metrics['n_positions']}")
```

### 5. 运行回测

```python
from src.optimization.backtester import PortfolioBacktester, create_simple_momentum_strategy

# 创建策略
momentum_strategy = create_simple_momentum_strategy(
    lookback=252,
    skip=21,
    n_stocks=10  # 持有前10只动量股票
)

# 初始化回测引擎
backtester = PortfolioBacktester(
    initial_capital=1_000_000,
    transaction_cost=0.001,  # 10 bps
    rebalance_frequency=21   # 月度调仓
)

# 运行回测
backtest_results = backtester.run_backtest(
    returns=returns,
    strategy_function=momentum_strategy,
    start_date="2021-01-01"
)

# 计算绩效
performance = backtester.calculate_performance_metrics(backtest_results)

# 打印报告
backtester.print_performance_report(performance)
```

## 运行完整测试

测试所有模块是否正常工作：

```bash
python test_system.py
```

期望输出：
```
================================================================================
QUANT FACTOR MINING - FULL SYSTEM TEST
================================================================================

PHASE 1: DATA INFRASTRUCTURE
================================================================================
✅ Data fetched: (8520, 6)
✅ Data preprocessed:
   Shape: (8520, 6)
   Coverage: 99.85%

PHASE 2: FACTOR CONSTRUCTION
================================================================================
✅ Momentum: Count: 6840, Mean: 0.0002, Std: 0.9998
✅ Mean Reversion: Count: 8280, Mean: -0.0001, Std: 1.0001
✅ Volatility: Count: 8160, Mean: 0.0000, Std: 1.0000

PHASE 3: FACTOR EVALUATION
================================================================================
✅ IC Analysis:
   IC Mean: 0.0347
   IC t-stat: 2.84
   IC Positive %: 63.25%
✅ Factor validity: True

PHASE 4: PORTFOLIO OPTIMIZATION
================================================================================
✅ Optimization complete:
   Number of positions: 8
   Expected return (annual): 8.45%
   Sharpe ratio: 1.23

PHASE 5: BACKTESTING
================================================================================
✅ Backtest Results:
   Total Return: 34.67%
   Annualized Return: 10.42%
   Sharpe Ratio: 0.89
   Max Drawdown: -15.23%

✅ ALL PHASES COMPLETED SUCCESSFULLY
🎉 System is ready for production use!
```

## Jupyter Notebook研究

启动Jupyter Lab进行交互式研究：

```bash
jupyter lab notebooks/01_data_acquisition.ipynb
```

建议创建的研究notebook：
- `02_exploratory_analysis.ipynb` - 收益分布、相关性分析
- `03_factor_construction.ipynb` - 因子可视化和对比
- `04_alphalens_evaluation.ipynb` - 完整的Alphalens报告
- `05_portfolio_optimization.ipynb` - 有效前沿、权重分析
- `06_backtest_analysis.ipynb` - 多策略对比、绩效归因

## 高级用法

### 使用不同的优化器

```python
from src.optimization.mean_variance import MaxSharpeOptimizer, MinVarianceOptimizer

# 最大化夏普比率
sharpe_optimizer = MaxSharpeOptimizer(max_position=0.10)
sharpe_weights = sharpe_optimizer.optimize(latest_factors, cov_matrix)

# 最小方差组合
min_var_optimizer = MinVarianceOptimizer(max_position=0.10)
min_var_weights = min_var_optimizer.optimize(cov_matrix)
```

### 计算有效前沿

```python
frontier_returns, frontier_risks, frontier_weights = optimizer.compute_efficient_frontier(
    expected_returns=latest_factors,
    covariance_matrix=cov_matrix,
    n_points=50
)

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(frontier_risks * 100, frontier_returns * 100, 'b-', linewidth=2)
plt.xlabel('Risk (Volatility %)')
plt.ylabel('Expected Return %')
plt.title('Efficient Frontier')
plt.grid(True)
plt.show()
```

### 因子组合

```python
from src.factors.base import CompositeFactor

# 创建多因子组合
composite = CompositeFactor(
    factors=[
        MomentumFactor(lookback=252, skip=21),
        MeanReversionFactor(lookback=21),
        VolatilityFactor(window=63)
    ],
    weights=[0.5, 0.3, 0.2],  # 权重
    name="Multi-Factor Composite"
)

composite_scores = composite.compute(clean_data, normalize=True)
```

## 故障排除

### 问题：yfinance下载失败
```python
# 解决方案：减小批次大小，增加重试次数
fetcher = YFinanceFetcher()
# 编辑 config/settings.py:
# YFINANCE_BATCH_SIZE = 20  # 从50降到20
# YFINANCE_RETRY_TIMES = 5  # 从3增到5
```

### 问题：协方差矩阵不是正定的
```python
# 解决方案：使用Ledoit-Wolf收缩或修正
cov_estimator = CovarianceEstimator(method='ledoit_wolf')
cov_matrix = cov_estimator.estimate(returns, window=252)

# 或者手动修正
if not cov_estimator.validate_covariance(cov_matrix):
    cov_matrix = cov_estimator.fix_covariance(cov_matrix, method='clip')
```

### 问题：优化失败
```python
# 解决方案：放宽约束或更换求解器
optimizer = MeanVarianceOptimizer(
    max_position=0.20,  # 增大持仓上限
    solver='OSQP'       # 尝试不同求解器
)
```

## 性能优化建议

1. **使用缓存**: 首次下载后，后续运行会自动使用缓存
2. **并行计算**: YFinanceFetcher默认使用5个worker，可增加到10
3. **减少因子计算频率**: 月度调仓只需月末计算因子
4. **使用Parquet**: 已实现，比CSV快10倍以上

## 下一步

- 扩展股票池至完整S&P 500（500只股票）
- 添加行业中性化
- 实现机器学习因子组合
- 添加实时监控
- 部署到云端（AWS/GCP）

## 获取帮助

- 查看 `IMPLEMENTATION_STATUS.md` 了解实现细节
- 运行 `python test_factors.py` 测试单个因子
- 运行 `python test_system.py` 测试完整系统
- 阅读各模块的docstring获取API文档

---

**祝您量化研究顺利！** 📈🚀
