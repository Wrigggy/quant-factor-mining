"""
风险模型模块
协方差矩阵估计：样本协方差、Ledoit-Wolf收缩、因子模型
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from loguru import logger

try:
    from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, some covariance estimators will not work")
    SKLEARN_AVAILABLE = False

from config.settings import COV_ESTIMATION_WINDOW, COV_ESTIMATION_METHOD


class CovarianceEstimator:
    """
    协方差矩阵估计器

    支持多种方法：
    1. 样本协方差（Sample Covariance）
    2. Ledoit-Wolf收缩（Ledoit-Wolf Shrinkage）
    3. Oracle Approximating Shrinkage (OAS)
    4. 因子模型协方差（Factor Model）
    5. 指数加权协方差（EWMA）
    """

    def __init__(self, method: str = COV_ESTIMATION_METHOD):
        """
        Parameters
        ----------
        method : str
            估计方法：'sample', 'ledoit_wolf', 'oas', 'factor_model', 'ewma'
        """
        self.method = method
        logger.info(f"CovarianceEstimator initialized with method: {method}")

    @staticmethod
    def sample_covariance(
        returns: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        样本协方差矩阵

        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据 (dates × assets)
        window : int, optional
            使用最近N天的数据，None表示使用全部

        Returns
        -------
        pd.DataFrame
            协方差矩阵 (assets × assets)
        """
        logger.info("Computing sample covariance matrix...")

        if window is not None:
            returns = returns.tail(window)

        cov_matrix = returns.cov()

        logger.info(f"Sample covariance computed: {cov_matrix.shape}")
        return cov_matrix

    @staticmethod
    def ledoit_wolf_shrinkage(
        returns: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Ledoit-Wolf收缩估计

        优点：减少估计误差，特别适用于高维数据（N > T）
        原理：在样本协方差和目标矩阵之间线性收缩

        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
        window : int, optional
            使用最近N天的数据

        Returns
        -------
        pd.DataFrame
            收缩后的协方差矩阵
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, falling back to sample covariance")
            return CovarianceEstimator.sample_covariance(returns, window)

        logger.info("Computing Ledoit-Wolf shrinkage covariance...")

        if window is not None:
            returns = returns.tail(window)

        # 去除缺失值
        returns_clean = returns.dropna()

        # Ledoit-Wolf估计
        lw = LedoitWolf()
        lw.fit(returns_clean)

        cov_matrix = pd.DataFrame(
            lw.covariance_,
            index=returns.columns,
            columns=returns.columns
        )

        logger.info(f"Ledoit-Wolf covariance computed: {cov_matrix.shape}")
        logger.info(f"Shrinkage intensity: {lw.shrinkage_:.4f}")

        return cov_matrix

    @staticmethod
    def oas_shrinkage(
        returns: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Oracle Approximating Shrinkage (OAS) 估计

        类似Ledoit-Wolf，但使用不同的收缩公式

        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
        window : int, optional
            使用最近N天的数据

        Returns
        -------
        pd.DataFrame
            收缩后的协方差矩阵
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, falling back to sample covariance")
            return CovarianceEstimator.sample_covariance(returns, window)

        logger.info("Computing OAS shrinkage covariance...")

        if window is not None:
            returns = returns.tail(window)

        returns_clean = returns.dropna()

        oas = OAS()
        oas.fit(returns_clean)

        cov_matrix = pd.DataFrame(
            oas.covariance_,
            index=returns.columns,
            columns=returns.columns
        )

        logger.info(f"OAS covariance computed: {cov_matrix.shape}")
        logger.info(f"Shrinkage intensity: {oas.shrinkage_:.4f}")

        return cov_matrix

    @staticmethod
    def ewma_covariance(
        returns: pd.DataFrame,
        span: int = 60,
        min_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        指数加权移动平均协方差（EWMA）

        给予近期数据更高权重

        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
        span : int
            EWMA span参数（默认60天=3个月）
        min_periods : int, optional
            最小观测数

        Returns
        -------
        pd.DataFrame
            EWMA协方差矩阵
        """
        logger.info(f"Computing EWMA covariance with span={span}...")

        min_periods = min_periods or span // 2

        # 计算EWMA协方差
        ewma_cov = returns.ewm(span=span, min_periods=min_periods).cov().iloc[-len(returns.columns):]

        logger.info(f"EWMA covariance computed: {ewma_cov.shape}")
        return ewma_cov

    @staticmethod
    def factor_model_covariance(
        returns: pd.DataFrame,
        factors: pd.DataFrame,
        window: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        因子模型协方差估计

        模型：Σ = B × F × B^T + D

        其中：
        - B = 因子载荷矩阵（beta）
        - F = 因子协方差矩阵
        - D = 特质风险对角矩阵

        Parameters
        ----------
        returns : pd.DataFrame
            资产收益率 (dates × assets)
        factors : pd.DataFrame
            因子收益率 (dates × factors)
        window : int, optional
            估计窗口

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (协方差矩阵, 因子载荷, 特质风险)
        """
        logger.info("Computing factor model covariance...")

        if window is not None:
            returns = returns.tail(window)
            factors = factors.tail(window)

        # 对齐数据
        common_dates = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_dates]
        factors_aligned = factors.loc[common_dates]

        # 回归计算因子载荷（beta）
        from sklearn.linear_model import LinearRegression

        n_assets = len(returns_aligned.columns)
        n_factors = len(factors_aligned.columns)

        factor_loadings = np.zeros((n_assets, n_factors))
        residuals = np.zeros((len(returns_aligned), n_assets))

        for i, asset in enumerate(returns_aligned.columns):
            y = returns_aligned[asset].values.reshape(-1, 1)
            X = factors_aligned.values

            # 去除NaN
            valid_idx = ~(np.isnan(y).flatten() | np.isnan(X).any(axis=1))
            y_clean = y[valid_idx]
            X_clean = X[valid_idx]

            if len(y_clean) > n_factors:
                model = LinearRegression()
                model.fit(X_clean, y_clean)

                factor_loadings[i, :] = model.coef_.flatten()
                residuals[valid_idx, i] = (y_clean - model.predict(X_clean)).flatten()

        # 转换为DataFrame
        B = pd.DataFrame(
            factor_loadings,
            index=returns_aligned.columns,
            columns=factors_aligned.columns
        )

        # 计算因子协方差
        F = factors_aligned.cov()

        # 计算特质风险（对角矩阵）
        residual_var = np.var(residuals, axis=0)
        D = pd.DataFrame(
            np.diag(residual_var),
            index=returns_aligned.columns,
            columns=returns_aligned.columns
        )

        # 完整协方差矩阵：Σ = B × F × B^T + D
        cov_matrix = B @ F @ B.T + D

        logger.info(f"Factor model covariance computed: {cov_matrix.shape}")
        logger.info(f"Factors: {n_factors}, Assets: {n_assets}")

        return cov_matrix, B, D

    def estimate(
        self,
        returns: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None,
        window: int = COV_ESTIMATION_WINDOW,
        **kwargs
    ) -> pd.DataFrame:
        """
        根据指定方法估计协方差矩阵

        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
        factors : pd.DataFrame, optional
            因子数据（仅factor_model方法需要）
        window : int
            估计窗口
        **kwargs
            传递给具体方法的额外参数

        Returns
        -------
        pd.DataFrame
            协方差矩阵
        """
        method = kwargs.get('method', self.method)

        logger.info(f"Estimating covariance with method: {method}, window: {window}")

        if method == 'sample':
            cov_matrix = self.sample_covariance(returns, window)

        elif method == 'ledoit_wolf':
            cov_matrix = self.ledoit_wolf_shrinkage(returns, window)

        elif method == 'oas':
            cov_matrix = self.oas_shrinkage(returns, window)

        elif method == 'ewma':
            span = kwargs.get('span', 60)
            cov_matrix = self.ewma_covariance(returns, span)

        elif method == 'factor_model':
            if factors is None:
                raise ValueError("factors required for factor_model method")
            cov_matrix, _, _ = self.factor_model_covariance(returns, factors, window)

        else:
            raise ValueError(f"Unknown covariance estimation method: {method}")

        if not self.validate_covariance(cov_matrix):
            cov_matrix = self.fix_covariance(cov_matrix, method='clip')

        return cov_matrix

    @staticmethod
    def validate_covariance(cov_matrix: pd.DataFrame) -> bool:
        """
        验证协方差矩阵是否有效

        检查：
        1. 对称性
        2. 正定性

        Parameters
        ----------
        cov_matrix : pd.DataFrame
            协方差矩阵

        Returns
        -------
        bool
            是否有效
        """
        # 检查对称性
        is_symmetric = np.allclose(cov_matrix, cov_matrix.T)

        # 检查正定性（所有特征值 > 0）
        eigenvalues = np.linalg.eigvals(cov_matrix.values)
        is_positive_definite = np.all(eigenvalues > 0)

        if not is_symmetric:
            logger.warning("Covariance matrix is not symmetric")

        if not is_positive_definite:
            logger.warning(f"Covariance matrix is not positive definite. Min eigenvalue: {eigenvalues.min():.6f}")

        return is_symmetric and is_positive_definite

    @staticmethod
    def fix_covariance(cov_matrix: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
        """
        修正无效的协方差矩阵

        Parameters
        ----------
        cov_matrix : pd.DataFrame
            协方差矩阵
        method : str
            修正方法：'clip'（截断负特征值）或'shrink'（收缩）

        Returns
        -------
        pd.DataFrame
            修正后的协方差矩阵
        """
        logger.info(f"Fixing covariance matrix using method: {method}")

        if method == 'clip':
            # 特征值分解
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix.values)

            # 截断负特征值
            eigenvalues_clipped = np.maximum(eigenvalues, 1e-8)

            # 重构协方差矩阵
            cov_fixed = eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T

            cov_fixed_df = pd.DataFrame(
                cov_fixed,
                index=cov_matrix.index,
                columns=cov_matrix.columns
            )

            logger.info(f"Clipped {(eigenvalues < 0).sum()} negative eigenvalues")
            return cov_fixed_df

        elif method == 'shrink':
            # 向单位矩阵收缩
            identity = np.eye(len(cov_matrix))
            shrinkage = 0.1
            cov_fixed = (1 - shrinkage) * cov_matrix.values + shrinkage * identity

            cov_fixed_df = pd.DataFrame(
                cov_fixed,
                index=cov_matrix.index,
                columns=cov_matrix.columns
            )

            logger.info(f"Applied shrinkage: {shrinkage}")
            return cov_fixed_df

        else:
            raise ValueError(f"Unknown fix method: {method}")


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher
    from src.data.preprocessor import DataPreprocessor
    from src.data.cache_manager import CacheManager

    logger.add("risk_models_test.log", rotation="10 MB")

    # 获取数据
    fetcher = YFinanceFetcher()
    cache = CacheManager()
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM"]

    data = cache.get_or_fetch("US", "test", test_tickers, "2020-01-01", "2024-01-01", fetcher)

    preprocessor = DataPreprocessor()
    clean_data, _ = preprocessor.validate_data_quality(data)

    # 计算收益率
    close = clean_data['close'].unstack(level='ticker')
    returns = close.pct_change().dropna()

    print(f"\nReturns shape: {returns.shape}")

    # 测试不同的协方差估计方法
    estimator = CovarianceEstimator()

    print("\n=== Sample Covariance ===")
    cov_sample = estimator.sample_covariance(returns, window=252)
    print(f"Shape: {cov_sample.shape}")
    print(f"Valid: {estimator.validate_covariance(cov_sample)}")
    print(f"\nSample:\n{cov_sample.iloc[:3, :3]}")

    print("\n=== Ledoit-Wolf Shrinkage ===")
    cov_lw = estimator.ledoit_wolf_shrinkage(returns, window=252)
    print(f"Shape: {cov_lw.shape}")
    print(f"Valid: {estimator.validate_covariance(cov_lw)}")
    print(f"\nSample:\n{cov_lw.iloc[:3, :3]}")

    print("\n=== EWMA Covariance ===")
    cov_ewma = estimator.ewma_covariance(returns, span=60)
    print(f"Shape: {cov_ewma.shape}")
    print(f"Valid: {estimator.validate_covariance(cov_ewma)}")

    # 比较不同方法
    print("\n=== Method Comparison ===")
    methods = ['sample', 'ledoit_wolf', 'ewma']
    for method in methods:
        cov = estimator.estimate(returns, window=252, method=method, span=60)
        print(f"{method}: Mean variance = {np.diag(cov.values).mean():.6f}")
