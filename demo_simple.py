"""
简化演示脚本 - 避免pandas版本兼容问题
展示核心功能：数据获取、因子计算、优化、回测
"""

import sys
sys.path.append('.')

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from loguru import logger

# 配置简洁输出
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

print("\n" + "="*80)
print("美股多因子量化投资系统 - 简化演示")
print("="*80)

# ==================== 第1步：数据获取 ====================
print("\n📥 第1步：数据获取与预处理")
print("-" * 80)

from config.universes import get_universe
from src.data.fetcher import YFinanceFetcher
from src.data.cache_manager import CacheManager
from src.data.preprocessor import DataPreprocessor

fetcher = YFinanceFetcher()
cache = CacheManager()
preprocessor = DataPreprocessor()

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
print(f"股票池: {tickers}")

print("正在获取数据...")
data = cache.get_or_fetch(
    market="US",
    universe="demo_simple",
    tickers=tickers,
    start_date="2020-01-01",
    end_date="2024-01-01",
    fetcher=fetcher
)

print(f"✅ 原始数据: {data.shape}")

clean_data, report = preprocessor.validate_data_quality(data)
print(f"✅ 清洗后数据: {report['final_shape']}")
print(f"   数据覆盖率: {report['data_coverage']:.2%}")

# ==================== 第2步：因子计算 ====================
print("\n🔢 第2步：计算Alpha因子")
print("-" * 80)

from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor

print("计算动量因子...")
momentum = MomentumFactor(lookback=252, skip=21)
momentum_scores = momentum.compute(clean_data, normalize=True)
print(f"✅ 动量因子: {len(momentum_scores)} 个观测值")

print("计算均值回归因子...")
mean_reversion = MeanReversionFactor(lookback=21)
reversal_scores = mean_reversion.compute(clean_data, normalize=True)
print(f"✅ 均值回归因子: {len(reversal_scores)} 个观测值")

print("计算波动率因子...")
volatility = VolatilityFactor(window=63)
vol_scores = volatility.compute(clean_data, normalize=True)
print(f"✅ 波动率因子: {len(vol_scores)} 个观测值")

# ==================== 第3步：简单统计分析 ====================
print("\n📊 第3步：因子统计分析")
print("-" * 80)

print("\n动量因子统计:")
print(f"  均值: {momentum_scores.mean():.4f}")
print(f"  标准差: {momentum_scores.std():.4f}")
print(f"  最小值: {momentum_scores.min():.4f}")
print(f"  最大值: {momentum_scores.max():.4f}")

print("\n最新因子得分 (前5):")
latest_momentum = momentum_scores.groupby(level='ticker').last().sort_values(ascending=False)
print(latest_momentum.head())

# ==================== 第4步：组合优化 ====================
print("\n🎯 第4步：投资组合优化")
print("-" * 80)

from src.optimization.risk_models import CovarianceEstimator
from src.optimization.mean_variance import MeanVarianceOptimizer

# 使用最新因子得分
latest_factors = momentum_scores.groupby(level='ticker').last()
print(f"使用最新因子得分: {len(latest_factors)} 只股票")

# 计算收益率和协方差
close = clean_data['close'].unstack(level='ticker')
returns = close.pct_change().dropna()

print("估计协方差矩阵（Ledoit-Wolf收缩）...")
cov_estimator = CovarianceEstimator(method='ledoit_wolf')
cov_matrix = cov_estimator.estimate(returns, window=252)
print(f"✅ 协方差矩阵: {cov_matrix.shape}")

print("运行CVXPY优化...")
optimizer = MeanVarianceOptimizer(
    risk_aversion=1.0,
    max_position=0.25,
    long_only=True
)

optimal_weights = optimizer.optimize(
    expected_returns=latest_factors,
    covariance_matrix=cov_matrix
)

print(f"\n✅ 优化完成!")
print(f"   持仓数量: {(optimal_weights > 1e-4).sum()}")
print(f"\n   投资组合权重:")
for ticker, weight in optimal_weights.sort_values(ascending=False).items():
    if weight > 1e-4:
        print(f"     {ticker}: {weight:.2%}")

# 计算组合指标
metrics = optimizer.compute_portfolio_metrics(optimal_weights, latest_factors, cov_matrix)
print(f"\n   组合指标:")
print(f"     预期年化收益: {metrics['expected_return_annual']:.2%}")
print(f"     年化波动率: {metrics['volatility_annual']:.2%}")
print(f"     夏普比率: {metrics['sharpe_ratio']:.4f}")

# ==================== 第5步：简单回测 ====================
print("\n🚀 第5步：策略回测")
print("-" * 80)

from src.optimization.backtester import PortfolioBacktester, create_simple_momentum_strategy

print("创建动量策略（持有前3只动量股票）...")
momentum_strategy = create_simple_momentum_strategy(
    lookback=252,
    skip=21,
    n_stocks=3
)

backtester = PortfolioBacktester(
    initial_capital=1_000_000,
    transaction_cost=0.001,
    rebalance_frequency=21
)

print(f"回测配置:")
print(f"  初始资金: ${backtester.initial_capital:,}")
print(f"  交易成本: {backtester.transaction_cost*10000:.1f} bps")
print(f"  调仓频率: {backtester.rebalance_frequency} 天")

print("\n运行回测（2021-2023，留1年warmup）...")
backtest_results = backtester.run_backtest(
    returns=returns,
    strategy_function=momentum_strategy,
    start_date="2021-01-01",
    end_date="2023-12-31"
)

performance = backtester.calculate_performance_metrics(backtest_results)

print("\n" + "="*80)
print("回测结果摘要")
print("="*80)

print(f"\n收益指标:")
print(f"  总收益率: {performance['total_return_pct']:.2f}%")
print(f"  年化收益: {performance['annualized_return_pct']:.2f}%")

print(f"\n风险指标:")
print(f"  年化波动率: {performance['volatility_annual_pct']:.2f}%")
print(f"  最大回撤: {performance['max_drawdown_pct']:.2f}%")

print(f"\n风险调整收益:")
print(f"  夏普比率: {performance['sharpe_ratio']:.4f}")
print(f"  索提诺比率: {performance['sortino_ratio']:.4f}")
print(f"  卡玛比率: {performance['calmar_ratio']:.4f}")

print(f"\n交易统计:")
print(f"  胜率: {performance['win_rate_pct']:.2f}%")
print(f"  平均换手率: {performance['avg_turnover_pct']:.2f}%")
print(f"  调仓次数: {performance['n_rebalances']}")

print(f"\n交易成本:")
print(f"  总成本: ${performance['total_txn_costs']:,.2f}")
print(f"  成本占比: {performance['txn_cost_pct']:.3f}%")

# ==================== 总结 ====================
print("\n" + "="*80)
print("✅ 演示完成！")
print("="*80)

print("\n📋 流程总结:")
print("  1. ✅ 数据获取与预处理")
print("  2. ✅ Alpha因子计算（3个因子）")
print("  3. ✅ 因子统计分析")
print("  4. ✅ 投资组合优化（CVXPY）")
print("  5. ✅ 策略回测与绩效评估")

print("\n🎯 关键结果:")
print(f"  股票数量: {len(tickers)}")
print(f"  最优组合夏普: {metrics['sharpe_ratio']:.4f}")
print(f"  回测总收益: {performance['total_return_pct']:.2f}%")
print(f"  回测夏普比率: {performance['sharpe_ratio']:.4f}")
print(f"  最大回撤: {performance['max_drawdown_pct']:.2f}%")

print("\n📚 更多功能:")
print("  - 查看 QUICKSTART.md 了解完整API")
print("  - 查看 PROJECT_SUMMARY.md 了解技术细节")
print("  - 运行 jupyter lab 进行交互式研究")
print("  - 访问 https://github.com/Wrigggy/quant-factor-mining")

print("\n🎉 感谢使用！\n")
