"""
投资组合回测引擎
模拟真实交易，计算绩效指标
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, Optional, List, Tuple
from datetime import datetime
from loguru import logger

from config.settings import (
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    REBALANCE_FREQUENCY,
    RISK_FREE_RATE
)


class PortfolioBacktester:
    """
    投资组合回测引擎

    功能：
    1. 模拟交易执行
    2. 计算交易成本
    3. 跟踪换手率
    4. 计算绩效指标（夏普、最大回撤等）
    5. 生成回测报告
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        transaction_cost: float = TRANSACTION_COST,
        rebalance_frequency: int = REBALANCE_FREQUENCY,
        risk_free_rate: float = RISK_FREE_RATE
    ):
        """
        Parameters
        ----------
        initial_capital : float
            初始资金（默认$1,000,000）
        transaction_cost : float
            交易成本（单边，默认0.001 = 10 bps）
        rebalance_frequency : int
            调仓频率（交易日，默认21天=月度）
        risk_free_rate : float
            无风险利率（年化，默认0.02 = 2%）
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate

        logger.info("PortfolioBacktester initialized:")
        logger.info(f"  Initial capital: ${initial_capital:,.0f}")
        logger.info(f"  Transaction cost: {transaction_cost:.4f} ({transaction_cost*10000:.1f} bps)")
        logger.info(f"  Rebalance frequency: {rebalance_frequency} days")
        logger.info(f"  Risk-free rate: {risk_free_rate:.2%}")

    def run_backtest(
        self,
        returns: pd.DataFrame,
        strategy_function: Callable,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **strategy_kwargs
    ) -> pd.DataFrame:
        """
        运行回测

        Parameters
        ----------
        returns : pd.DataFrame
            资产收益率 (dates × assets)
        strategy_function : Callable
            策略函数，接受(returns_history, current_date, **kwargs)，返回目标权重
        start_date : str, optional
            回测开始日期
        end_date : str, optional
            回测结束日期
        **strategy_kwargs
            传递给策略函数的参数

        Returns
        -------
        pd.DataFrame
            回测结果，包含日期、权重、收益、资金等
        """
        logger.info("Starting backtest...")

        # 确定回测区间
        trading_dates = returns.index
        if start_date:
            trading_dates = trading_dates[trading_dates >= start_date]
        if end_date:
            trading_dates = trading_dates[trading_dates <= end_date]

        logger.info(f"Backtest period: {trading_dates[0]} to {trading_dates[-1]}")
        logger.info(f"Total trading days: {len(trading_dates)}")

        # 初始化
        portfolio_value = self.initial_capital
        current_weights = pd.Series(0, index=returns.columns)  # 初始权重为0（现金）

        # 结果记录
        results = []

        # 调仓日期
        rebalance_dates = trading_dates[::self.rebalance_frequency]
        logger.info(f"Rebalance dates: {len(rebalance_dates)}")

        for i, date in enumerate(trading_dates):
            # 检查是否需要调仓
            if date in rebalance_dates:
                logger.debug(f"Rebalancing on {date}")

                # 调用策略函数获取目标权重
                try:
                    returns_history = returns.loc[:date]
                    target_weights = strategy_function(
                        returns_history,
                        date,
                        **strategy_kwargs
                    )

                    # 对齐资产
                    target_weights = target_weights.reindex(returns.columns, fill_value=0)

                except Exception as e:
                    logger.error(f"Strategy function failed on {date}: {e}")
                    target_weights = current_weights  # 保持不变

                # 计算换手率
                turnover = (target_weights - current_weights).abs().sum()

                # 计算交易成本
                txn_costs = turnover * self.transaction_cost * portfolio_value
                portfolio_value -= txn_costs

                # 更新权重
                current_weights = target_weights

                logger.debug(f"  Turnover: {turnover:.2%}, TXN costs: ${txn_costs:,.2f}")

            else:
                turnover = 0
                txn_costs = 0
                target_weights = current_weights

            # 计算当日收益
            daily_returns = returns.loc[date]
            portfolio_return = (current_weights * daily_returns).sum()

            # 更新资金
            portfolio_value *= (1 + portfolio_return)

            # 权重漂移（由于价格变动）
            current_weights = current_weights * (1 + daily_returns)
            current_weights = current_weights / current_weights.sum() if current_weights.sum() != 0 else current_weights

            # 记录结果
            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'portfolio_return': portfolio_return,
                'turnover': turnover,
                'transaction_costs': txn_costs,
                'is_rebalance': date in rebalance_dates
            })

            # 进度日志
            if (i + 1) % 252 == 0:  # 每年
                logger.info(f"Progress: {i+1}/{len(trading_dates)} days, "
                           f"Value: ${portfolio_value:,.0f}, "
                           f"Return: {(portfolio_value / self.initial_capital - 1):.2%}")

        results_df = pd.DataFrame(results).set_index('date')

        logger.info("✅ Backtest complete")
        logger.info(f"Final portfolio value: ${portfolio_value:,.0f}")
        logger.info(f"Total return: {(portfolio_value / self.initial_capital - 1):.2%}")

        return results_df

    def calculate_performance_metrics(
        self,
        backtest_results: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        计算绩效指标

        Parameters
        ----------
        backtest_results : pd.DataFrame
            回测结果
        benchmark_returns : pd.Series, optional
            基准收益率（用于计算alpha和beta）

        Returns
        -------
        Dict
            绩效指标字典
        """
        logger.info("Calculating performance metrics...")

        returns = backtest_results['portfolio_return']
        portfolio_values = backtest_results['portfolio_value']

        # 基本收益指标
        total_return = (portfolio_values.iloc[-1] / self.initial_capital) - 1
        n_days = len(returns)
        n_years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1

        # 风险指标
        volatility_daily = returns.std()
        volatility_annual = volatility_daily * np.sqrt(252)

        # 夏普比率
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        # 索提诺比率（只考虑下行风险）
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 and len(downside_returns) > 0 else 0

        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 卡玛比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 胜率
        win_rate = (returns > 0).mean()

        # 盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).sum() > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).sum() > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0

        # 交易成本分析
        total_turnover = backtest_results['turnover'].sum()
        avg_turnover = backtest_results[backtest_results['turnover'] > 0]['turnover'].mean()
        total_txn_costs = backtest_results['transaction_costs'].sum()
        txn_cost_pct = total_txn_costs / self.initial_capital

        metrics = {
            # 收益指标
            'total_return': total_return,
            'annualized_return': annualized_return,
            'total_return_pct': total_return * 100,
            'annualized_return_pct': annualized_return * 100,

            # 风险指标
            'volatility_daily': volatility_daily,
            'volatility_annual': volatility_annual,
            'volatility_annual_pct': volatility_annual * 100,

            # 风险调整收益
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,

            # 回撤
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,

            # 交易统计
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_loss_ratio': profit_loss_ratio,

            # 换手率
            'total_turnover': total_turnover,
            'avg_turnover': avg_turnover,
            'avg_turnover_pct': avg_turnover * 100,
            'n_rebalances': backtest_results['is_rebalance'].sum(),

            # 交易成本
            'total_txn_costs': total_txn_costs,
            'txn_cost_pct': txn_cost_pct * 100,
            'txn_costs_vs_return': txn_cost_pct / total_return if total_return != 0 else 0,

            # 时间统计
            'n_trading_days': n_days,
            'n_years': n_years,
        }

        # 基准比较（如果提供）
        if benchmark_returns is not None:
            aligned_returns = returns.align(benchmark_returns, join='inner')
            portfolio_returns_aligned = aligned_returns[0]
            benchmark_returns_aligned = aligned_returns[1]

            # Alpha和Beta
            covariance = np.cov(portfolio_returns_aligned, benchmark_returns_aligned)[0, 1]
            benchmark_variance = benchmark_returns_aligned.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0

            benchmark_mean = benchmark_returns_aligned.mean()
            alpha_daily = portfolio_returns_aligned.mean() - beta * benchmark_mean
            alpha_annual = alpha_daily * 252

            # 信息比率
            active_returns = portfolio_returns_aligned - benchmark_returns_aligned
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0

            metrics.update({
                'alpha_annual': alpha_annual,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
            })

        logger.info("Performance metrics calculated")
        return metrics

    def calculate_drawdown_series(self, backtest_results: pd.DataFrame) -> pd.Series:
        """
        计算回撤时间序列

        Parameters
        ----------
        backtest_results : pd.DataFrame
            回测结果

        Returns
        -------
        pd.Series
            回撤序列
        """
        portfolio_values = backtest_results['portfolio_value']
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max

        return drawdown

    def print_performance_report(self, metrics: Dict):
        """
        打印绩效报告

        Parameters
        ----------
        metrics : Dict
            绩效指标字典
        """
        print("\n" + "="*60)
        print("PORTFOLIO PERFORMANCE REPORT")
        print("="*60)

        print("\n--- Return Metrics ---")
        print(f"Total Return:        {metrics['total_return_pct']:>10.2f}%")
        print(f"Annualized Return:   {metrics['annualized_return_pct']:>10.2f}%")

        print("\n--- Risk Metrics ---")
        print(f"Annual Volatility:   {metrics['volatility_annual_pct']:>10.2f}%")
        print(f"Max Drawdown:        {metrics['max_drawdown_pct']:>10.2f}%")

        print("\n--- Risk-Adjusted Returns ---")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.4f}")
        print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.4f}")
        print(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.4f}")

        print("\n--- Trading Statistics ---")
        print(f"Win Rate:            {metrics['win_rate_pct']:>10.2f}%")
        print(f"Profit/Loss Ratio:   {metrics['profit_loss_ratio']:>10.4f}")
        print(f"Avg Turnover:        {metrics['avg_turnover_pct']:>10.2f}%")
        print(f"Number of Rebalances:{metrics['n_rebalances']:>10.0f}")

        print("\n--- Transaction Costs ---")
        print(f"Total TXN Costs:     ${metrics['total_txn_costs']:>10,.2f}")
        print(f"TXN Costs (% of AUM):{metrics['txn_cost_pct']:>10.4f}%")
        print(f"TXN Costs vs Return: {metrics['txn_costs_vs_return']*100:>10.2f}%")

        if 'alpha_annual' in metrics:
            print("\n--- Benchmark Comparison ---")
            print(f"Alpha (Annual):      {metrics['alpha_annual']*100:>10.4f}%")
            print(f"Beta:                {metrics['beta']:>10.4f}")
            print(f"Information Ratio:   {metrics['information_ratio']:>10.4f}")
            print(f"Tracking Error:      {metrics['tracking_error']*100:>10.2f}%")

        print("\n--- Period ---")
        print(f"Trading Days:        {metrics['n_trading_days']:>10.0f}")
        print(f"Years:               {metrics['n_years']:>10.2f}")

        print("="*60 + "\n")


def create_simple_momentum_strategy(
    lookback: int = 252,
    skip: int = 21,
    n_stocks: int = 10
):
    """
    创建简单的动量策略

    策略逻辑：
    1. 计算过去N个月的收益率
    2. 选择表现最好的K只股票
    3. 等权配置

    Parameters
    ----------
    lookback : int
        回看期
    skip : int
        跳过天数
    n_stocks : int
        持仓股票数

    Returns
    -------
    Callable
        策略函数
    """
    def strategy(returns_history, current_date, **kwargs):
        """策略函数"""

        # 确保有足够的历史数据
        if len(returns_history) < lookback + skip:
            # 返回等权组合
            return pd.Series(1.0 / len(returns_history.columns), index=returns_history.columns)

        # 计算动量得分
        start_idx = -(lookback + skip)
        end_idx = -skip if skip > 0 else None

        momentum_returns = returns_history.iloc[start_idx:end_idx].sum()

        # 选择top K
        top_stocks = momentum_returns.nlargest(n_stocks)

        # 等权配置
        weights = pd.Series(0, index=returns_history.columns)
        weights[top_stocks.index] = 1.0 / len(top_stocks)

        return weights

    return strategy


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher
    from src.data.preprocessor import DataPreprocessor
    from src.data.cache_manager import CacheManager

    logger.add("backtester_test.log", rotation="10 MB")

    # 获取数据
    fetcher = YFinanceFetcher()
    cache = CacheManager()
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM", "JNJ", "WMT"]

    data = cache.get_or_fetch("US", "test", test_tickers, "2018-01-01", "2024-01-01", fetcher)

    preprocessor = DataPreprocessor()
    clean_data, _ = preprocessor.validate_data_quality(data)

    # 计算收益率
    close = clean_data['close'].unstack(level='ticker')
    returns = close.pct_change().dropna()

    print(f"\nReturns shape: {returns.shape}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")

    # 创建简单动量策略
    momentum_strategy = create_simple_momentum_strategy(lookback=252, skip=21, n_stocks=5)

    # 运行回测
    backtester = PortfolioBacktester(
        initial_capital=1_000_000,
        transaction_cost=0.001,  # 10 bps
        rebalance_frequency=21   # 月度
    )

    backtest_results = backtester.run_backtest(
        returns=returns,
        strategy_function=momentum_strategy,
        start_date="2019-01-01"  # 留1年作为warmup
    )

    # 计算绩效指标
    metrics = backtester.calculate_performance_metrics(backtest_results)

    # 打印报告
    backtester.print_performance_report(metrics)

    # 保存结果
    print("\n=== Backtest Results Sample ===")
    print(backtest_results.head(10))
    print(f"\nTotal records: {len(backtest_results)}")
