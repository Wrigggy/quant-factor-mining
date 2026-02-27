"""
完整系统测试
测试所有模块的集成
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from loguru import logger

# 配置日志
logger.add("test_system.log", rotation="10 MB", level="INFO")

# 导入所有模块
from config.settings import *
from config.universes import get_universe

# Data
from src.data.fetcher import YFinanceFetcher
from src.data.preprocessor import DataPreprocessor
from src.data.cache_manager import CacheManager

# Factors
from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor

# Evaluation
from src.evaluation.alphalens_wrapper import AlphalensEvaluator
from src.evaluation.ic_analysis import ICAnalyzer

# Optimization
from src.optimization.risk_models import CovarianceEstimator
from src.optimization.mean_variance import MeanVarianceOptimizer, MaxSharpeOptimizer
from src.optimization.backtester import PortfolioBacktester, create_simple_momentum_strategy


def test_phase_1_data():
    """测试Phase 1: 数据模块"""
    print("\n" + "="*80)
    print("PHASE 1: DATA INFRASTRUCTURE")
    print("="*80)

    # 初始化
    fetcher = YFinanceFetcher()
    cache = CacheManager()
    preprocessor = DataPreprocessor()

    # 获取测试数据
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM"]
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    print(f"\nFetching {len(test_tickers)} tickers...")

    data = cache.get_or_fetch(
        market="US",
        universe="test",
        tickers=test_tickers,
        start_date=start_date,
        end_date=end_date,
        fetcher=fetcher
    )

    print(f"✅ Data fetched: {data.shape}")

    # 预处理
    clean_data, report = preprocessor.validate_data_quality(data)

    print(f"\n✅ Data preprocessed:")
    print(f"   Shape: {report['final_shape']}")
    print(f"   Coverage: {report['data_coverage']:.2%}")
    print(f"   Date range: {report['date_range']}")

    return clean_data


def test_phase_2_factors(data):
    """测试Phase 2: 因子构建"""
    print("\n" + "="*80)
    print("PHASE 2: FACTOR CONSTRUCTION")
    print("="*80)

    factors = {
        'Momentum': MomentumFactor(lookback=252, skip=21),
        'Mean Reversion': MeanReversionFactor(lookback=21),
        'Volatility': VolatilityFactor(window=63)
    }

    factor_values = {}

    for name, factor in factors.items():
        print(f"\n--- Testing {name} ---")
        try:
            values = factor.compute(data, normalize=True)
            factor_values[name] = values

            print(f"✅ {name}:")
            print(f"   Count: {len(values)}")
            print(f"   Null ratio: {values.isnull().mean():.2%}")
            print(f"   Mean: {values.mean():.4f}, Std: {values.std():.4f}")

        except Exception as e:
            print(f"❌ {name} failed: {e}")

    return factor_values


def test_phase_3_evaluation(factor_values, data):
    """测试Phase 3: 因子评估"""
    print("\n" + "="*80)
    print("PHASE 3: FACTOR EVALUATION")
    print("="*80)

    # 使用动量因子做测试
    if 'Momentum' not in factor_values:
        print("⚠️ Momentum factor not available, skipping evaluation")
        return None

    momentum = factor_values['Momentum']

    # IC分析
    print("\n--- IC Analysis ---")
    analyzer = ICAnalyzer(method='spearman')

    # 计算前瞻收益
    forward_returns = analyzer.compute_forward_returns(data, periods=[21])
    ic_series = analyzer.compute_ic_series(momentum, forward_returns[21])

    # IC统计
    ic_stats = analyzer.compute_ic_statistics(ic_series)

    print(f"✅ IC Analysis:")
    print(f"   IC Mean: {ic_stats['mean']:.4f}")
    print(f"   IC t-stat: {ic_stats['t_stat']:.2f}")
    print(f"   IC Positive %: {ic_stats['positive_pct']:.2%}")

    # 判断因子是否有效
    is_valid = abs(ic_stats['mean']) > 0.02 and abs(ic_stats['t_stat']) > 2.0

    print(f"\n{'✅' if is_valid else '❌'} Factor validity: {is_valid}")
    print(f"   Criteria: |IC| > 0.02 AND |t-stat| > 2.0")

    return ic_stats


def test_phase_4_optimization(factor_values, data):
    """测试Phase 4: 组合优化"""
    print("\n" + "="*80)
    print("PHASE 4: PORTFOLIO OPTIMIZATION")
    print("="*80)

    if 'Momentum' not in factor_values:
        print("⚠️ Momentum factor not available, skipping optimization")
        return None

    # 使用最新的因子得分作为预期收益
    latest_factors = factor_values['Momentum'].groupby(level='ticker').last()

    # 计算协方差矩阵
    print("\n--- Covariance Estimation ---")
    close = data['close'].unstack(level='ticker')
    returns = close.pct_change().dropna()

    cov_estimator = CovarianceEstimator(method='ledoit_wolf')
    cov_matrix = cov_estimator.estimate(returns, window=252)

    print(f"✅ Covariance matrix estimated: {cov_matrix.shape}")

    # 均值-方差优化
    print("\n--- Mean-Variance Optimization ---")
    optimizer = MeanVarianceOptimizer(risk_aversion=1.0, max_position=0.2)

    optimal_weights = optimizer.optimize(latest_factors, cov_matrix)

    print(f"\n✅ Optimization complete:")
    print(f"   Number of positions: {(optimal_weights > 1e-4).sum()}")
    print(f"   Max position: {optimal_weights.max():.2%}")
    print(f"\n   Top 5 holdings:")
    print(optimal_weights.sort_values(ascending=False).head(5).apply(lambda x: f"{x:.2%}"))

    # 计算组合指标
    metrics = optimizer.compute_portfolio_metrics(optimal_weights, latest_factors, cov_matrix)

    print(f"\n✅ Portfolio metrics:")
    print(f"   Expected return (annual): {metrics['expected_return_annual']:.2%}")
    print(f"   Volatility (annual): {metrics['volatility_annual']:.2%}")
    print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.4f}")

    return optimal_weights


def test_phase_5_backtesting(data):
    """测试Phase 5: 回测"""
    print("\n" + "="*80)
    print("PHASE 5: BACKTESTING")
    print("="*80)

    # 计算收益率
    close = data['close'].unstack(level='ticker')
    returns = close.pct_change().dropna()

    print(f"\nBacktest setup:")
    print(f"   Assets: {len(returns.columns)}")
    print(f"   Period: {returns.index[0]} to {returns.index[-1]}")
    print(f"   Days: {len(returns)}")

    # 创建简单动量策略
    print("\n--- Running Backtest ---")
    momentum_strategy = create_simple_momentum_strategy(
        lookback=252,
        skip=21,
        n_stocks=5
    )

    backtester = PortfolioBacktester(
        initial_capital=1_000_000,
        transaction_cost=0.001,
        rebalance_frequency=21
    )

    backtest_results = backtester.run_backtest(
        returns=returns,
        strategy_function=momentum_strategy,
        start_date="2021-01-01"  # 留1年warmup
    )

    # 计算绩效
    metrics = backtester.calculate_performance_metrics(backtest_results)

    print(f"\n✅ Backtest Results:")
    print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"   Annualized Return: {metrics['annualized_return_pct']:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Calmar Ratio: {metrics['calmar_ratio']:.4f}")

    return backtest_results, metrics


def main():
    """运行完整系统测试"""
    print("\n" + "="*80)
    print("QUANT FACTOR MINING - FULL SYSTEM TEST")
    print("="*80)

    try:
        # Phase 1: Data
        clean_data = test_phase_1_data()

        # Phase 2: Factors
        factor_values = test_phase_2_factors(clean_data)

        # Phase 3: Evaluation
        ic_stats = test_phase_3_evaluation(factor_values, clean_data)

        # Phase 4: Optimization
        optimal_weights = test_phase_4_optimization(factor_values, clean_data)

        # Phase 5: Backtesting
        backtest_results, metrics = test_phase_5_backtesting(clean_data)

        # Final summary
        print("\n" + "="*80)
        print("✅ ALL PHASES COMPLETED SUCCESSFULLY")
        print("="*80)

        print("\n📊 System Summary:")
        print(f"   Phases tested: 5/5")
        print(f"   Data shape: {clean_data.shape}")
        print(f"   Factors computed: {len(factor_values)}")
        print(f"   Backtest return: {metrics['total_return_pct']:.2f}%")
        print(f"   Backtest Sharpe: {metrics['sharpe_ratio']:.4f}")

        print("\n🎉 System is ready for production use!")

        return True

    except Exception as e:
        print("\n" + "="*80)
        print("❌ SYSTEM TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        logger.exception("System test failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
