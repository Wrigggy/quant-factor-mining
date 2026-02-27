"""
均值-方差投资组合优化器
使用CVXPY进行凸优化
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from loguru import logger

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    logger.error("cvxpy not installed. Install with: pip install cvxpy")
    CVXPY_AVAILABLE = False

from config.settings import (
    RISK_AVERSION,
    MAX_POSITION_SIZE,
    TURNOVER_LIMIT,
    LONG_ONLY
)


class MeanVarianceOptimizer:
    """
    Markowitz均值-方差投资组合优化器

    目标函数：
        maximize: w^T μ - λ × w^T Σ w

    约束条件：
        - Σ w_i = 1                     (满仓)
        - w_i >= 0                      (仅多头，可选)
        - w_i <= max_position          (持仓上限)
        - Σ |w_i - w_prev| <= turnover  (换手率限制，可选)
    """

    def __init__(
        self,
        risk_aversion: float = RISK_AVERSION,
        max_position: float = MAX_POSITION_SIZE,
        long_only: bool = LONG_ONLY,
        solver: str = 'ECOS'
    ):
        """
        Parameters
        ----------
        risk_aversion : float
            风险厌恶系数 λ（默认1.0）
        max_position : float
            单只股票最大持仓比例（默认0.05 = 5%）
        long_only : bool
            是否仅多头（默认True）
        solver : str
            CVXPY求解器：'ECOS', 'SCS', 'OSQP'等
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy is required. Install with: pip install cvxpy")

        self.risk_aversion = risk_aversion
        self.max_position = max_position
        self.long_only = long_only
        self.solver = solver

        logger.info(f"MeanVarianceOptimizer initialized:")
        logger.info(f"  Risk aversion: {risk_aversion}")
        logger.info(f"  Max position: {max_position:.2%}")
        logger.info(f"  Long only: {long_only}")
        logger.info(f"  Solver: {solver}")

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        prev_weights: Optional[pd.Series] = None,
        turnover_limit: Optional[float] = None,
        min_weight: float = 0.0,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None
    ) -> pd.Series:
        """
        优化投资组合权重

        Parameters
        ----------
        expected_returns : pd.Series
            预期收益率向量（通常使用因子得分作为代理）
        covariance_matrix : pd.DataFrame
            协方差矩阵
        prev_weights : pd.Series, optional
            前期权重（用于换手率约束）
        turnover_limit : float, optional
            换手率上限（例如0.5表示50%）
        min_weight : float
            最小持仓比例（默认0）
        target_return : float, optional
            目标收益率（如果指定，则优化变为最小化风险）
        target_risk : float, optional
            目标风险水平（如果指定，则优化变为最大化收益）

        Returns
        -------
        pd.Series
            最优权重，indexed by asset
        """
        logger.info("Starting portfolio optimization...")

        # 对齐数据
        assets = expected_returns.index.intersection(covariance_matrix.index)
        mu = expected_returns.loc[assets].values
        Sigma = covariance_matrix.loc[assets, assets].values

        n_assets = len(assets)
        logger.info(f"Optimizing {n_assets} assets")

        # 定义决策变量
        w = cp.Variable(n_assets)

        # 目标函数
        if target_return is not None:
            # 目标收益约束下最小化风险
            objective = cp.Minimize(cp.quad_form(w, Sigma))
        elif target_risk is not None:
            # 目标风险约束下最大化收益
            objective = cp.Maximize(mu @ w)
        else:
            # 标准均值-方差优化
            portfolio_return = mu @ w
            portfolio_variance = cp.quad_form(w, Sigma)
            objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_variance)

        # 约束条件
        constraints = []

        # 1. 满仓约束
        constraints.append(cp.sum(w) == 1)

        # 2. 多空约束
        if self.long_only:
            constraints.append(w >= min_weight)
        else:
            constraints.append(w >= -self.max_position)  # 做空上限

        # 3. 持仓上限
        constraints.append(w <= self.max_position)

        # 4. 换手率约束
        if turnover_limit is not None and prev_weights is not None:
            w_prev = prev_weights.reindex(assets, fill_value=0).values
            # L1范数（曼哈顿距离）代表换手率
            constraints.append(cp.norm(w - w_prev, 1) <= turnover_limit)
            logger.info(f"Turnover limit: {turnover_limit:.2%}")

        # 5. 目标收益约束
        if target_return is not None:
            constraints.append(mu @ w >= target_return)
            logger.info(f"Target return: {target_return:.4f}")

        # 6. 目标风险约束
        if target_risk is not None:
            constraints.append(cp.quad_form(w, Sigma) <= target_risk ** 2)
            logger.info(f"Target risk: {target_risk:.4f}")

        # 构建优化问题
        problem = cp.Problem(objective, constraints)

        # 求解
        try:
            problem.solve(solver=self.solver, verbose=False)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                logger.error(f"Optimization failed with status: {problem.status}")
                raise ValueError(f"Optimization failed: {problem.status}")

            # 提取最优权重
            optimal_weights = pd.Series(w.value, index=assets)

            # 四舍五入（避免极小权重）
            optimal_weights = optimal_weights.round(6)

            # 验证结果
            self._validate_solution(optimal_weights, expected_returns, covariance_matrix)

            logger.info("✅ Optimization successful")
            logger.info(f"Portfolio return: {(optimal_weights * expected_returns).sum():.4f}")
            logger.info(f"Portfolio risk: {np.sqrt(optimal_weights @ covariance_matrix @ optimal_weights):.4f}")
            logger.info(f"Number of positions: {(optimal_weights.abs() > 1e-4).sum()}")

            return optimal_weights

        except Exception as e:
            logger.error(f"Optimization error: {e}")
            raise

    def _validate_solution(
        self,
        weights: pd.Series,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ):
        """验证优化解的有效性"""

        # 检查权重和
        weight_sum = weights.sum()
        if not np.isclose(weight_sum, 1.0, atol=1e-3):
            logger.warning(f"Weight sum: {weight_sum:.6f} (expected 1.0)")

        # 检查持仓上限
        max_weight = weights.abs().max()
        if max_weight > self.max_position + 1e-4:
            logger.warning(f"Max weight: {max_weight:.4f} exceeds limit {self.max_position:.4f}")

        # 检查多空约束
        if self.long_only:
            min_weight = weights.min()
            if min_weight < -1e-4:
                logger.warning(f"Negative weight found: {min_weight:.6f} (long_only=True)")

    def compute_efficient_frontier(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Series]]:
        """
        计算有效前沿

        Parameters
        ----------
        expected_returns : pd.Series
            预期收益
        covariance_matrix : pd.DataFrame
            协方差矩阵
        n_points : int
            前沿上的点数

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[pd.Series]]
            (收益数组, 风险数组, 权重列表)
        """
        logger.info(f"Computing efficient frontier with {n_points} points...")

        # 计算最小/最大可行收益
        min_return = expected_returns.min()
        max_return = expected_returns.max()

        target_returns = np.linspace(min_return, max_return, n_points)

        frontier_returns = []
        frontier_risks = []
        frontier_weights = []

        for target_return in target_returns:
            try:
                # 在目标收益下最小化风险
                weights = self.optimize(
                    expected_returns,
                    covariance_matrix,
                    target_return=target_return
                )

                portfolio_return = (weights * expected_returns).sum()
                portfolio_risk = np.sqrt(weights @ covariance_matrix @ weights)

                frontier_returns.append(portfolio_return)
                frontier_risks.append(portfolio_risk)
                frontier_weights.append(weights)

            except Exception as e:
                logger.warning(f"Failed to compute point at target return {target_return:.4f}: {e}")
                continue

        logger.info(f"Computed {len(frontier_returns)} points on efficient frontier")

        return np.array(frontier_returns), np.array(frontier_risks), frontier_weights

    def compute_portfolio_metrics(
        self,
        weights: pd.Series,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        计算投资组合的各项指标

        Parameters
        ----------
        weights : pd.Series
            权重
        expected_returns : pd.Series
            预期收益
        covariance_matrix : pd.DataFrame
            协方差矩阵
        risk_free_rate : float
            无风险利率（年化）

        Returns
        -------
        Dict
            指标字典
        """
        # 对齐数据
        assets = weights.index
        mu = expected_returns.loc[assets]
        Sigma = covariance_matrix.loc[assets, assets]

        # 计算指标
        portfolio_return = (weights * mu).sum()
        portfolio_variance = weights @ Sigma @ weights
        portfolio_std = np.sqrt(portfolio_variance)

        # 夏普比率（年化）
        sharpe_ratio = (portfolio_return * 252 - risk_free_rate) / (portfolio_std * np.sqrt(252))

        # 集中度（Herfindahl指数）
        concentration = (weights ** 2).sum()

        # 有效持仓数
        effective_n = 1 / concentration

        metrics = {
            'expected_return_daily': portfolio_return,
            'expected_return_annual': portfolio_return * 252,
            'volatility_daily': portfolio_std,
            'volatility_annual': portfolio_std * np.sqrt(252),
            'sharpe_ratio': sharpe_ratio,
            'concentration': concentration,
            'effective_n_stocks': effective_n,
            'n_positions': (weights.abs() > 1e-4).sum(),
            'max_weight': weights.abs().max(),
            'long_exposure': weights[weights > 0].sum(),
            'short_exposure': -weights[weights < 0].sum(),
        }

        return metrics


class MaxSharpeOptimizer(MeanVarianceOptimizer):
    """
    最大化夏普比率优化器

    目标：maximize (w^T μ - r_f) / sqrt(w^T Σ w)
    """

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
        **kwargs
    ) -> pd.Series:
        """
        最大化夏普比率

        转换为标准二次规划：
        1. 令 y = w / (w^T μ - r_f)
        2. 最小化 y^T Σ y
        3. 约束 y^T (μ - r_f) = 1
        """
        logger.info("Optimizing for maximum Sharpe ratio...")

        assets = expected_returns.index.intersection(covariance_matrix.index)
        mu = expected_returns.loc[assets].values
        Sigma = covariance_matrix.loc[assets, assets].values
        r_f = risk_free_rate / 252  # 日度无风险利率

        n_assets = len(assets)

        # 决策变量 y
        y = cp.Variable(n_assets)

        # 目标：最小化 y^T Σ y
        objective = cp.Minimize(cp.quad_form(y, Sigma))

        # 约束：y^T (μ - r_f) = 1
        constraints = [
            (mu - r_f) @ y == 1,
            y >= 0  # 仅多头
        ]

        # 求解
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed: {problem.status}")

        # 转换回权重 w = y / sum(y)
        y_opt = y.value
        w_opt = y_opt / y_opt.sum()

        optimal_weights = pd.Series(w_opt, index=assets)

        logger.info("✅ Max Sharpe optimization successful")
        metrics = self.compute_portfolio_metrics(
            optimal_weights,
            expected_returns,
            covariance_matrix,
            risk_free_rate
        )
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")

        return optimal_weights


class MinVarianceOptimizer(MeanVarianceOptimizer):
    """
    最小方差优化器

    目标：minimize w^T Σ w
    """

    def optimize(
        self,
        covariance_matrix: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        最小化投资组合方差
        """
        logger.info("Optimizing for minimum variance...")

        assets = covariance_matrix.index
        Sigma = covariance_matrix.values
        n_assets = len(assets)

        w = cp.Variable(n_assets)

        # 目标：最小化方差
        objective = cp.Minimize(cp.quad_form(w, Sigma))

        # 约束
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.max_position
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed: {problem.status}")

        optimal_weights = pd.Series(w.value, index=assets)

        logger.info("✅ Min variance optimization successful")
        portfolio_risk = np.sqrt(optimal_weights @ covariance_matrix @ optimal_weights)
        logger.info(f"Portfolio risk: {portfolio_risk:.4f}")

        return optimal_weights


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher
    from src.data.preprocessor import DataPreprocessor
    from src.data.cache_manager import CacheManager
    from src.factors.momentum import MomentumFactor
    from src.optimization.risk_models import CovarianceEstimator

    logger.add("mean_variance_test.log", rotation="10 MB")

    # 获取数据
    fetcher = YFinanceFetcher()
    cache = CacheManager()
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM"]

    data = cache.get_or_fetch("US", "test", test_tickers, "2020-01-01", "2024-01-01", fetcher)

    preprocessor = DataPreprocessor()
    clean_data, _ = preprocessor.validate_data_quality(data)

    # 计算因子作为预期收益的代理
    momentum = MomentumFactor(lookback=252, skip=21)
    factor_values = momentum.compute(clean_data, normalize=True)

    # 使用最新的因子得分
    latest_factors = factor_values.groupby(level='ticker').last()

    # 估计协方差矩阵
    close = clean_data['close'].unstack(level='ticker')
    returns = close.pct_change().dropna()

    cov_estimator = CovarianceEstimator(method='ledoit_wolf')
    cov_matrix = cov_estimator.estimate(returns, window=252)

    print("\n=== Mean-Variance Optimization ===")
    optimizer = MeanVarianceOptimizer(risk_aversion=1.0, max_position=0.2)

    optimal_weights = optimizer.optimize(latest_factors, cov_matrix)

    print(f"\nOptimal Weights:")
    print(optimal_weights.sort_values(ascending=False))

    print(f"\nPortfolio Metrics:")
    metrics = optimizer.compute_portfolio_metrics(optimal_weights, latest_factors, cov_matrix)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n=== Max Sharpe Optimization ===")
    sharpe_optimizer = MaxSharpeOptimizer(max_position=0.2)
    sharpe_weights = sharpe_optimizer.optimize(latest_factors, cov_matrix)

    print(f"\nMax Sharpe Weights:")
    print(sharpe_weights.sort_values(ascending=False))
