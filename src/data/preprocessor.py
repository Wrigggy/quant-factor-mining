"""
数据预处理模块
处理复权、缺失值、异常值、Winsorize等
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from config.settings import (
    MAX_MISSING_RATIO,
    MAX_CONSECUTIVE_MISSING,
    MAX_SINGLE_DAY_RETURN,
    WINSORIZE_LOWER,
    WINSORIZE_UPPER
)


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self):
        logger.info("DataPreprocessor initialized")

    def adjust_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        使用Adj Close计算调整因子，应用到OHLC

        Parameters
        ----------
        data : pd.DataFrame
            原始数据，包含 'close', 'adj close', 'open', 'high', 'low'

        Returns
        -------
        pd.DataFrame
            复权后的数据
        """
        logger.info("Adjusting prices for stock splits and dividends")

        df = data.copy()

        # 计算调整因子
        if 'adj close' in df.columns and 'close' in df.columns:
            df['adj_factor'] = df['adj close'] / df['close']

            # 应用到OHLC
            for col in ['open', 'high', 'low']:
                if col in df.columns:
                    df[col] = df[col] * df['adj_factor']

            # 使用adj close作为close
            df['close'] = df['adj close']

            # 删除不需要的列
            df = df.drop(columns=['adj close', 'adj_factor'], errors='ignore')

        logger.info("Price adjustment completed")
        return df

    def handle_missing_values(
        self,
        data: pd.DataFrame,
        method: str = "ffill",
        max_consecutive: int = MAX_CONSECUTIVE_MISSING
    ) -> pd.DataFrame:
        """
        处理缺失值

        Parameters
        ----------
        data : pd.DataFrame
            输入数据
        method : str
            填充方法：'ffill' (前向填充), 'bfill' (后向填充), 'interpolate' (插值)
        max_consecutive : int
            最大连续缺失天数

        Returns
        -------
        pd.DataFrame
            处理后的数据
        """
        logger.info(f"Handling missing values with method: {method}")

        df = data.copy()
        initial_nulls = df.isnull().sum().sum()

        # 按ticker分组处理
        if method == "ffill":
            df = df.groupby(level='ticker').ffill(limit=max_consecutive)
        elif method == "bfill":
            df = df.groupby(level='ticker').bfill(limit=max_consecutive)
        elif method == "interpolate":
            # 对数值列进行线性插值
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df.groupby(level='ticker')[numeric_cols].apply(
                lambda x: x.interpolate(method='linear', limit=max_consecutive)
            )

        final_nulls = df.isnull().sum().sum()
        logger.info(f"Missing values: {initial_nulls} -> {final_nulls} (filled {initial_nulls - final_nulls})")

        return df

    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        检测并处理异常值

        异常检测规则：
        1. 负价格 -> NaN
        2. High < Low -> 删除该行
        3. 单日涨跌 > 100% -> 标记为异常

        Parameters
        ----------
        data : pd.DataFrame
            输入数据

        Returns
        -------
        pd.DataFrame
            处理后的数据
        """
        logger.info("Detecting and handling anomalies")

        df = data.copy()
        anomaly_count = 0

        # 1. 负价格检测
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                negative_mask = df[col] < 0
                negative_count = negative_mask.sum()
                if negative_count > 0:
                    df.loc[negative_mask, col] = np.nan
                    anomaly_count += negative_count
                    logger.warning(f"Found {negative_count} negative values in '{col}'")

        # 2. High < Low 检测
        if 'high' in df.columns and 'low' in df.columns:
            invalid_mask = df['high'] < df['low']
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                df.loc[invalid_mask, price_cols] = np.nan
                anomaly_count += invalid_count
                logger.warning(f"Found {invalid_count} rows where High < Low")

        # 3. 极端收益率检测（按ticker分组）
        if 'close' in df.columns:
            # 计算日收益率
            df['_return'] = df.groupby(level='ticker')['close'].pct_change()

            # 检测极端收益
            extreme_mask = df['_return'].abs() > MAX_SINGLE_DAY_RETURN
            extreme_count = extreme_mask.sum()
            if extreme_count > 0:
                # 标记但不删除，让后续步骤处理
                logger.warning(f"Found {extreme_count} extreme returns (>{MAX_SINGLE_DAY_RETURN:.0%})")
                # 可选：设置为NaN
                # df.loc[extreme_mask, 'close'] = np.nan

            df = df.drop(columns=['_return'])

        logger.info(f"Total anomalies detected and handled: {anomaly_count}")
        return df

    def winsorize_data(
        self,
        data: pd.DataFrame,
        lower: float = WINSORIZE_LOWER,
        upper: float = WINSORIZE_UPPER,
        by_ticker: bool = False
    ) -> pd.DataFrame:
        """
        Winsorize处理（截尾）

        Parameters
        ----------
        data : pd.DataFrame
            输入数据
        lower : float
            下分位数（例如0.01表示1%）
        upper : float
            上分位数（例如0.99表示99%）
        by_ticker : bool
            是否按ticker分别处理

        Returns
        -------
        pd.DataFrame
            处理后的数据
        """
        logger.info(f"Winsorizing data: [{lower:.2%}, {upper:.2%}]")

        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if by_ticker:
            # 按ticker分组winsorize
            for col in numeric_cols:
                df[col] = df.groupby(level='ticker')[col].transform(
                    lambda x: x.clip(lower=x.quantile(lower), upper=x.quantile(upper))
                )
        else:
            # 全局winsorize
            for col in numeric_cols:
                lower_bound = df[col].quantile(lower)
                upper_bound = df[col].quantile(upper)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        logger.info("Winsorization completed")
        return df

    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        综合数据质量验证

        Parameters
        ----------
        data : pd.DataFrame
            输入数据

        Returns
        -------
        pd.DataFrame
            清洗后的数据
        Dict
            数据质量报告
        """
        logger.info("Starting comprehensive data quality validation")

        report = {
            'initial_shape': data.shape,
            'initial_nulls': data.isnull().sum().sum(),
            'tickers_count': len(data.index.get_level_values('ticker').unique()),
            'date_range': (
                data.index.get_level_values('date').min(),
                data.index.get_level_values('date').max()
            ),
            'anomalies': {},
            'final_shape': None,
            'final_nulls': None
        }

        # 按ticker统计缺失值比例
        missing_by_ticker = data.groupby(level='ticker').apply(
            lambda x: x.isnull().sum().sum() / (len(x) * len(x.columns))
        )
        bad_tickers = missing_by_ticker[missing_by_ticker > MAX_MISSING_RATIO].index.tolist()

        if bad_tickers:
            logger.warning(f"Removing {len(bad_tickers)} tickers with >50% missing data")
            report['removed_tickers'] = bad_tickers
            data = data.drop(bad_tickers, level='ticker')

        # 应用所有预处理步骤
        data = self.adjust_prices(data)
        data = self.detect_anomalies(data)
        data = self.handle_missing_values(data)

        # 最终统计
        report['final_shape'] = data.shape
        report['final_nulls'] = data.isnull().sum().sum()
        report['data_coverage'] = 1 - (report['final_nulls'] / (data.shape[0] * data.shape[1]))

        logger.info("Data quality validation completed")
        logger.info(f"Shape: {report['initial_shape']} -> {report['final_shape']}")
        logger.info(f"Nulls: {report['initial_nulls']} -> {report['final_nulls']}")
        logger.info(f"Data coverage: {report['data_coverage']:.2%}")

        return data, report

    def compute_returns(self, data: pd.DataFrame, periods: list = [1, 5, 21]) -> pd.DataFrame:
        """
        计算多期收益率

        Parameters
        ----------
        data : pd.DataFrame
            价格数据
        periods : list
            收益率周期（交易日）

        Returns
        -------
        pd.DataFrame
            包含收益率的数据
        """
        logger.info(f"Computing returns for periods: {periods}")

        df = data.copy()

        for period in periods:
            col_name = f'return_{period}d'
            df[col_name] = df.groupby(level='ticker')['close'].pct_change(period)

        # 对数收益率（用于波动率计算）
        df['log_return'] = df.groupby(level='ticker')['close'].apply(
            lambda x: np.log(x / x.shift(1))
        )

        logger.info("Returns computation completed")
        return df


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher

    # 配置日志
    logger.add("data_preprocessor.log", rotation="10 MB")

    # 获取测试数据
    fetcher = YFinanceFetcher()
    test_tickers = ["AAPL", "MSFT", "TSLA"]
    data = fetcher.fetch_batch(test_tickers, "2023-01-01", "2024-01-01")

    # 测试预处理
    preprocessor = DataPreprocessor()
    clean_data, report = preprocessor.validate_data_quality(data)

    print("\n=== Data Quality Report ===")
    for key, value in report.items():
        print(f"{key}: {value}")

    print("\n=== Clean Data Sample ===")
    print(clean_data.head(10))
