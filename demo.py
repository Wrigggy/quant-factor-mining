"""
完整工作流演示
展示从数据获取到回测的完整流程
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from loguru import logger

# 配置简洁的日志输出
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

print("\n" + "="*80)
print("美股/港股多因子量化投资系统 - 完整工作流演示")
print("="*80)

# ==================== 第1步：数据获取 ====================
print("\n📥 第1步：数据获取与预处理")
print("-" * 80)

from config.universes import get_universe
from src.data.fetcher import YFinanceFetcher
from src.data.cache_manager import CacheManager
from src.data.preprocessor import DataPreprocessor

# 初始化
fetcher = YFinanceFetcher()
cache = CacheManager()
preprocessor = DataPreprocessor()

# 选择股票池（示例：10只美股）
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "JNJ", "WMT"]
print(f"股票池: {tickers}")

# 获取数据
print("正在从yfinance获取数据...")
data = cache.get_or_fetch(
    market="US",
    universe="demo",
    tickers=tickers,
    start_date="2020-01-01",
    end_date="2024-01-01",
    fetcher=fetcher
)

print(f"✅ 原始数据: {data.shape}")

# 预处理
print("正在清洗数据...")
clean_data, report = preprocessor.validate_data_quality(data)

print(f"✅ 清洗后数据: {report['final_shape']}")
print(f"   数据覆盖率: {report['data_coverage']:.2%}")
print(f"   日期范围: {report['date_range'][0]} 至 {report['date_range'][1]}")

# ==================== 第2步：因子计算 ====================
print("\n🔢 第2步：计算Alpha因子")
print("-" * 80)

from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor

# 动量因子
print("计算动量因子（12个月动量，跳过1个月）...")
momentum = MomentumFactor(lookback=252, skip=21)
momentum_scores = momentum.compute(clean_data, normalize=True)
print(f"✅ 动量因子: {len(momentum_scores)} 个观测值, 均值={momentum_scores.mean():.4f}, 标准差={momentum_scores.std():.4f}")

# 均值回归因子
print("计算均值回归因子（1个月反转）...")
mean_reversion = MeanReversionFactor(lookback=21)
reversal_scores = mean_reversion.compute(clean_data, normalize=True)
print(f"✅ 均值回归因子: {len(reversal_scores)} 个观测值")

# 波动率因子
print("计算波动率因子（低波动率异象）...")
volatility = VolatilityFactor(window=63)
vol_scores = volatility.compute(clean_data, normalize=True)
print(f"✅ 波动率因子: {len(vol_scores)} 个观测值")

# ==================== 第3步：因子评估 ====================
print("\n📊 第3步：评估因子有效性（使用动量因子）")
print("-" * 80)

from src.evaluation.ic_analysis import ICAnalyzer

# IC分析
analyzer = ICAnalyzer(method='spearman')

print("计算IC统计量...")
forward_returns = analyzer.compute_forward_returns(clean_data, periods=[21])
ic_series = analyzer.compute_ic_series(momentum_scores, forward_returns[21])
ic_stats = analyzer.compute_ic_statistics(ic_series)

print(f"✅ IC分析结果:")
print(f"   IC均值: {ic_stats['mean']:.4f} (目标 > 0.02)")
print(f"   IC t统计量: {ic_stats['t_stat']:.2f} (目标 > 2.0)")
print(f"   IC正值占比: {ic_stats['positive_pct']:.2%} (目标 > 60%)")

# 判断因子是否有效
is_valid = abs(ic_stats['mean']) > 0.02 and abs(ic_stats['t_stat']) > 2.0
status = "✅ 有效" if is_valid else "❌ 无效"
print(f"\n因子评估: {status}")

# ==================== 第4步：组合优化 ====================
print("\n🎯 第4步：构建最优投资组合")
print("-" * 80)

from src.optimization.risk_models import CovarianceEstimator
from src.optimization.mean_variance import MeanVarianceOptimizer

# 使用最新的因子得分
latest_factors = momentum_scores.groupby(level='ticker').last()
print(f"使用最新因子得分: {len(latest_factors)} 只股票")

# 估计协方差矩阵
print("估计协方差矩阵（Ledoit-Wolf收缩）...")
close = clean_data['close'].unstack(level='ticker')
returns = close.pct_change().dropna()

cov_estimator = CovarianceEstimator(method='ledoit_wolf')
cov_matrix = cov_estimator.estimate(returns, window=252)
print(f"✅ 协方差矩阵: {cov_matrix.shape}")

# 优化投资组合
print("运行CVXPY优化器...")
optimizer = MeanVarianceOptimizer(
    risk_aversion=1.0,
    max_position=0.15,  # 单只股票最大15%
    long_only=True
)

optimal_weights = optimizer.optimize(
    expected_returns=latest_factors,
    covariance_matrix=cov_matrix
)

print(f"✅ 优化完成:")
print(f"   持仓数量: {(optimal_weights > 1e-4).sum()}")
print(f"\n   前5大持仓:")
for ticker, weight in optimal_weights.sort_values(ascending=False).head(5).items():
    print(f"     {ticker}: {weight:.2%}")

# 计算组合指标
metrics = optimizer.compute_portfolio_metrics(optimal_weights, latest_factors, cov_matrix)
print(f"\n   预期年化收益: {metrics['expected_return_annual']:.2%}")
print(f"   年化波动率: {metrics['volatility_annual']:.2%}")
print(f"   夏普比率: {metrics['sharpe_ratio']:.4f}")

# ==================== 第5步：策略回测 ====================
print("\n🚀 第5步：策略回测")
print("-" * 80)

from src.optimization.backtester import PortfolioBacktester, create_simple_momentum_strategy

# 创建简单动量策略
print("创建动量策略（持有前5只动量股票）...")
momentum_strategy = create_simple_momentum_strategy(
    lookback=252,
    skip=21,
    n_stocks=5
)

# 初始化回测引擎
backtester = PortfolioBacktester(
    initial_capital=1_000_000,
    transaction_cost=0.001,
    rebalance_frequency=21
)

print(f"回测配置:")
print(f"  初始资金: ${backtester.initial_capital:,}")
print(f"  交易成本: {backtester.transaction_cost*10000:.1f} bps")
print(f"  调仓频率: {backtester.rebalance_frequency} 天（月度）")

# 运行回测
print("\n运行回测...")
backtest_results = backtester.run_backtest(
    returns=returns,
    strategy_function=momentum_strategy,
    start_date="2021-01-01"  # 留1年warmup
)

# 计算绩效
performance = backtester.calculate_performance_metrics(backtest_results)

# 打印完整报告
print("\n" + "="*80)
backtester.print_performance_report(performance)

# ==================== 总结 ====================
print("\n" + "="*80)
print("✅ 完整工作流演示完成！")
print("="*80)

print("\n📋 流程总结:")
print("  1. ✅ 数据获取与预处理")
print("  2. ✅ Alpha因子计算（12个因子可选）")
print("  3. ✅ 因子有效性评估（IC分析）")
print("  4. ✅ 投资组合优化（CVXPY）")
print("  5. ✅ 策略回测与绩效评估")

print("\n🎯 关键结果:")
print(f"  因子有效性: {status}")
print(f"  组合夏普比率: {metrics['sharpe_ratio']:.4f}")
print(f"  回测总收益: {performance['total_return_pct']:.2f}%")
print(f"  回测年化收益: {performance['annualized_return_pct']:.2f}%")
print(f"  回测夏普比率: {performance['sharpe_ratio']:.4f}")
print(f"  最大回撤: {performance['max_drawdown_pct']:.2f}%")

print("\n📚 下一步:")
print("  - 运行 'python test_system.py' 测试完整系统")
print("  - 查看 QUICKSTART.md 了解更多用法")
print("  - 探索 notebooks/01_data_acquisition.ipynb")
print("  - 尝试不同的因子组合和策略")

print("\n🎉 感谢使用本系统，祝您量化研究顺利！\n")
