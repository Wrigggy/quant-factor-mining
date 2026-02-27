"""
因子基类
定义统一的因子接口和标准化方法
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List
import pandas as pd
import numpy as np
from loguru import logger

from config.settings import FACTOR_NORMALIZATION_METHOD


class BaseFactor(ABC):
    """
    Alpha因子抽象基类

    所有具体因子都应继承此类并实现calculate方法
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name : str, optional
            因子名称，默认使用类名
        """
        self.name = name or self.__class__.__name__
        logger.info(f"Initialized factor: {self.name}")

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值

        Parameters
        ----------
        data : pd.DataFrame
            输入数据，MultiIndex (date, ticker)，包含OHLCV列

        Returns
        -------
        pd.Series
            因子值，MultiIndex (date, ticker)
        """
        pass

    def normalize(
        self,
        factor_values: pd.Series,
        method: str = FACTOR_NORMALIZATION_METHOD,
        clip_std: Optional[float] = None
    ) -> pd.Series:
        """
        因子标准化（截面标准化）

        Parameters
        ----------
        factor_values : pd.Series
            原始因子值，MultiIndex (date, ticker)
        method : str
            标准化方法：
            - 'zscore': Z-Score标准化（截面）
            - 'rank': 百分位排名
            - 'minmax': 最小-最大标准化
        clip_std : float, optional
            截断标准差倍数（例如3表示[-3σ, 3σ]）

        Returns
        -------
        pd.Series
            标准化后的因子值
        """
        logger.debug(f"Normalizing factor with method: {method}")

        if method == "zscore":
            # 截面Z-Score标准化
            normalized = factor_values.groupby(level='date').apply(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )

            # 可选：截断极端值
            if clip_std is not None:
                normalized = normalized.clip(-clip_std, clip_std)

        elif method == "rank":
            # 截面百分位排名
            normalized = factor_values.groupby(level='date').rank(pct=True)

        elif method == "minmax":
            # 截面Min-Max标准化到[0, 1]
            normalized = factor_values.groupby(level='date').apply(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
            )

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def neutralize(
        self,
        factor_values: pd.Series,
        industry: Optional[pd.Series] = None,
        market_cap: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        因子中性化（去除行业和市值暴露）

        Parameters
        ----------
        factor_values : pd.Series
            因子值
        industry : pd.Series, optional
            行业分类，MultiIndex (date, ticker)
        market_cap : pd.Series, optional
            市值，MultiIndex (date, ticker)

        Returns
        -------
        pd.Series
            中性化后的因子值
        """
        logger.debug("Neutralizing factor")

        neutralized = factor_values.copy()

        # 按日期分组进行中性化
        def neutralize_cross_section(df):
            from sklearn.linear_model import LinearRegression

            y = df['factor'].values.reshape(-1, 1)
            X = []

            # 添加市值
            if 'market_cap' in df.columns:
                X.append(np.log(df['market_cap'].values + 1e-9).reshape(-1, 1))

            # 添加行业哑变量
            if 'industry' in df.columns:
                industry_dummies = pd.get_dummies(df['industry'])
                X.append(industry_dummies.values)

            if not X:
                return df['factor']

            X = np.hstack(X)

            # 回归并提取残差
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)

            return pd.Series(residuals.flatten(), index=df.index)

        # 构建DataFrame
        df = pd.DataFrame({'factor': factor_values})
        if industry is not None:
            df['industry'] = industry
        if market_cap is not None:
            df['market_cap'] = market_cap

        # 按日期分组中性化
        neutralized = df.groupby(level='date').apply(neutralize_cross_section)

        return neutralized

    def validate(self, factor_values: pd.Series) -> dict:
        """
        验证因子值的基本统计特性

        Parameters
        ----------
        factor_values : pd.Series
            因子值

        Returns
        -------
        dict
            验证报告
        """
        report = {
            'total_values': len(factor_values),
            'null_count': factor_values.isnull().sum(),
            'null_ratio': factor_values.isnull().mean(),
            'mean': factor_values.mean(),
            'std': factor_values.std(),
            'min': factor_values.min(),
            'max': factor_values.max(),
            'skewness': factor_values.skew(),
            'kurtosis': factor_values.kurtosis(),
            'unique_dates': len(factor_values.index.get_level_values('date').unique()),
            'unique_tickers': len(factor_values.index.get_level_values('ticker').unique()),
        }

        logger.info(f"Factor validation for {self.name}:")
        logger.info(f"  Total values: {report['total_values']}")
        logger.info(f"  Null ratio: {report['null_ratio']:.2%}")
        logger.info(f"  Mean: {report['mean']:.4f}, Std: {report['std']:.4f}")
        logger.info(f"  Range: [{report['min']:.4f}, {report['max']:.4f}]")

        return report

    def compute(
        self,
        data: pd.DataFrame,
        normalize: bool = True,
        normalization_method: Optional[str] = None,
        clip_std: Optional[float] = 3.0,
        industry: Optional[pd.Series] = None,
        market_cap: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        计算因子的完整流程

        Parameters
        ----------
        data : pd.DataFrame
            输入数据
        normalize : bool
            是否标准化
        normalization_method : str, optional
            标准化方法，默认使用配置
        clip_std : float, optional
            截断标准差倍数
        industry : pd.Series, optional
            行业分类（用于中性化）
        market_cap : pd.Series, optional
            市值（用于中性化）

        Returns
        -------
        pd.Series
            处理后的因子值
        """
        logger.info(f"Computing factor: {self.name}")

        # 1. 计算原始因子值
        factor_values = self.calculate(data)

        # 2. 验证
        self.validate(factor_values)

        # 3. 中性化（如果提供了行业或市值）
        if industry is not None or market_cap is not None:
            factor_values = self.neutralize(factor_values, industry, market_cap)

        # 4. 标准化
        if normalize:
            method = normalization_method or FACTOR_NORMALIZATION_METHOD
            factor_values = self.normalize(factor_values, method, clip_std)

        # 5. 删除NaN
        initial_count = len(factor_values)
        factor_values = factor_values.dropna()
        dropped_count = initial_count - len(factor_values)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} NaN values ({dropped_count/initial_count:.2%})")

        logger.info(f"Factor {self.name} computed successfully")
        return factor_values

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class CompositeFactor(BaseFactor):
    """
    复合因子：多个因子的加权组合
    """

    def __init__(
        self,
        factors: List[BaseFactor],
        weights: Optional[List[float]] = None,
        name: str = "CompositeFactor"
    ):
        """
        Parameters
        ----------
        factors : List[BaseFactor]
            因子列表
        weights : List[float], optional
            因子权重，默认等权
        name : str
            复合因子名称
        """
        super().__init__(name)
        self.factors = factors
        self.weights = weights or [1.0 / len(factors)] * len(factors)

        if len(self.factors) != len(self.weights):
            raise ValueError("Number of factors must match number of weights")

        if not np.isclose(sum(self.weights), 1.0):
            logger.warning(f"Weights sum to {sum(self.weights)}, normalizing to 1.0")
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

        logger.info(f"CompositeFactor initialized with {len(factors)} factors")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算复合因子值（加权平均）

        Parameters
        ----------
        data : pd.DataFrame
            输入数据

        Returns
        -------
        pd.Series
            复合因子值
        """
        factor_values_list = []

        for factor, weight in zip(self.factors, self.weights):
            logger.debug(f"Computing {factor.name} (weight: {weight})")
            values = factor.calculate(data)
            # 标准化单个因子
            values = factor.normalize(values)
            factor_values_list.append(values * weight)

        # 加权求和
        composite = pd.concat(factor_values_list, axis=1).sum(axis=1)

        logger.info(f"CompositeFactor computed from {len(self.factors)} factors")
        return composite


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher
    from src.data.preprocessor import DataPreprocessor

    # 配置日志
    logger.add("factor_base.log", rotation="10 MB")

    # 创建简单的测试因子
    class SimpleReturnFactor(BaseFactor):
        def __init__(self, period=21):
            super().__init__(name=f"Return_{period}D")
            self.period = period

        def calculate(self, data: pd.DataFrame) -> pd.Series:
            close = data['close'].unstack(level='ticker')
            returns = close.pct_change(self.period)
            return returns.stack(dropna=False)

    # 获取测试数据
    fetcher = YFinanceFetcher()
    data = fetcher.fetch_batch(["AAPL", "MSFT", "GOOGL"], "2023-01-01", "2024-01-01")
    preprocessor = DataPreprocessor()
    clean_data, _ = preprocessor.validate_data_quality(data)

    # 测试因子
    factor = SimpleReturnFactor(period=21)
    factor_values = factor.compute(clean_data, normalize=True)

    print("\n=== Factor Values Sample ===")
    print(factor_values.head(20))
    print(f"\nFactor shape: {factor_values.shape}")
