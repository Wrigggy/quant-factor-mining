"""
YFinance数据获取模块
实现速率限制、批量处理、并行下载、重试机制
"""

import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm

from config.settings import (
    YFINANCE_RATE_LIMIT,
    YFINANCE_BATCH_SIZE,
    YFINANCE_MAX_WORKERS,
    YFINANCE_RETRY_TIMES,
    YFINANCE_RETRY_DELAY,
    MAX_MISSING_RATIO
)


class RateLimiter:
    """速率限制器"""

    def __init__(self, max_requests_per_hour: int):
        self.max_requests = max_requests_per_hour
        self.requests = []
        self.window = 3600  # 1小时（秒）

    def wait_if_needed(self):
        """如果达到速率限制，等待"""
        now = time.time()

        # 移除1小时前的请求记录
        self.requests = [req_time for req_time in self.requests if now - req_time < self.window]

        if len(self.requests) >= self.max_requests:
            # 计算需要等待的时间
            oldest_request = min(self.requests)
            wait_time = self.window - (now - oldest_request) + 1
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.requests = []

        self.requests.append(now)


class YFinanceFetcher:
    """YFinance数据获取器"""

    def __init__(self, rate_limit: Optional[int] = None):
        """
        Parameters
        ----------
        rate_limit : int, optional
            每小时最大请求数，默认使用配置文件中的值
        """
        self.rate_limiter = RateLimiter(rate_limit or YFINANCE_RATE_LIMIT)
        logger.info(f"YFinanceFetcher initialized with rate limit: {rate_limit or YFINANCE_RATE_LIMIT} req/hour")

    def fetch_single_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        retry_times: int = YFINANCE_RETRY_TIMES
    ) -> Optional[pd.DataFrame]:
        """
        获取单只股票的数据

        Parameters
        ----------
        ticker : str
            股票代码
        start_date : str
            开始日期 (YYYY-MM-DD)
        end_date : str
            结束日期 (YYYY-MM-DD)
        retry_times : int
            重试次数

        Returns
        -------
        pd.DataFrame or None
            包含OHLCV数据的DataFrame，失败返回None
        """
        self.rate_limiter.wait_if_needed()

        for attempt in range(retry_times):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date, auto_adjust=False)

                if df.empty:
                    logger.warning(f"{ticker}: No data returned")
                    return None

                # 检查数据质量
                missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
                if missing_ratio > MAX_MISSING_RATIO:
                    logger.warning(f"{ticker}: Too many missing values ({missing_ratio:.1%})")
                    return None

                # 添加Ticker列
                df['Ticker'] = ticker
                df = df.reset_index()

                # 标准化列名
                df = df.rename(columns={'Date': 'date'})
                df.columns = [col.lower() for col in df.columns]

                logger.debug(f"{ticker}: Successfully fetched {len(df)} rows")
                return df

            except Exception as e:
                logger.warning(f"{ticker}: Attempt {attempt + 1}/{retry_times} failed: {e}")
                if attempt < retry_times - 1:
                    time.sleep(YFINANCE_RETRY_DELAY)
                else:
                    logger.error(f"{ticker}: All retry attempts failed")
                    return None

        return None

    def fetch_batch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        max_workers: int = YFINANCE_MAX_WORKERS
    ) -> pd.DataFrame:
        """
        并行批量获取多只股票数据

        Parameters
        ----------
        tickers : List[str]
            股票代码列表
        start_date : str
            开始日期 (YYYY-MM-DD)
        end_date : str
            结束日期 (YYYY-MM-DD)
        max_workers : int
            并行线程数

        Returns
        -------
        pd.DataFrame
            合并后的数据，MultiIndex (date, ticker)
        """
        logger.info(f"Fetching {len(tickers)} tickers from {start_date} to {end_date}")

        results = []
        failed_tickers = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_ticker = {
                executor.submit(self.fetch_single_ticker, ticker, start_date, end_date): ticker
                for ticker in tickers
            }

            # 使用tqdm显示进度
            with tqdm(total=len(tickers), desc="Downloading", unit="ticker") as pbar:
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        df = future.result()
                        if df is not None:
                            results.append(df)
                        else:
                            failed_tickers.append(ticker)
                    except Exception as e:
                        logger.error(f"{ticker}: Unexpected error: {e}")
                        failed_tickers.append(ticker)
                    finally:
                        pbar.update(1)

        if failed_tickers:
            logger.warning(f"Failed to fetch {len(failed_tickers)} tickers: {failed_tickers[:10]}...")

        if not results:
            logger.error("No data fetched for any ticker")
            return pd.DataFrame()

        # 合并所有数据
        combined_df = pd.concat(results, ignore_index=True)

        # 转换为MultiIndex
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.set_index(['date', 'ticker'])

        # 选择需要的列
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume', 'adj close']
        combined_df = combined_df[[col for col in columns_to_keep if col in combined_df.columns]]

        logger.info(f"Successfully fetched data: {len(results)}/{len(tickers)} tickers, "
                   f"{len(combined_df)} total rows")

        return combined_df

    def fetch_in_batches(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        batch_size: int = YFINANCE_BATCH_SIZE
    ) -> pd.DataFrame:
        """
        分批获取大量股票数据

        Parameters
        ----------
        tickers : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        batch_size : int
            每批股票数量

        Returns
        -------
        pd.DataFrame
            合并后的完整数据
        """
        all_data = []
        num_batches = (len(tickers) + batch_size - 1) // batch_size

        logger.info(f"Fetching {len(tickers)} tickers in {num_batches} batches")

        for i in range(0, len(tickers), batch_size):
            batch_num = i // batch_size + 1
            batch_tickers = tickers[i:i + batch_size]

            logger.info(f"Processing batch {batch_num}/{num_batches}")

            batch_data = self.fetch_batch(batch_tickers, start_date, end_date)
            if not batch_data.empty:
                all_data.append(batch_data)

            # 批次间短暂延迟
            if i + batch_size < len(tickers):
                time.sleep(2)

        if not all_data:
            logger.error("No data fetched from any batch")
            return pd.DataFrame()

        # 合并所有批次
        final_df = pd.concat(all_data)
        logger.info(f"Total data fetched: {len(final_df)} rows")

        return final_df


def get_valid_date_range(start_date: str, end_date: str) -> tuple:
    """
    验证并返回有效的日期范围

    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns
    -------
    tuple
        (start_date, end_date) 字符串元组
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    today = pd.Timestamp.now()

    if start >= end:
        raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")

    if end > today:
        logger.warning(f"End date ({end_date}) is in the future, using today's date")
        end = today

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


if __name__ == "__main__":
    # 测试代码
    from config.universes import get_universe

    # 配置日志
    logger.add("data_fetcher.log", rotation="10 MB")

    # 测试获取少量股票
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    start = "2023-01-01"
    end = "2024-01-01"

    fetcher = YFinanceFetcher()
    data = fetcher.fetch_batch(test_tickers, start, end, max_workers=3)

    print("\n=== Data Summary ===")
    print(f"Shape: {data.shape}")
    print(f"\nFirst few rows:\n{data.head(10)}")
    print(f"\nData info:\n{data.info()}")
