"""
本地数据缓存管理模块
使用Parquet格式存储，避免重复API调用
"""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import pandas as pd
from loguru import logger

from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_EXPIRY_DAYS,
    CACHE_FORMAT,
    CACHE_COMPRESSION
)


class CacheManager:
    """缓存管理器"""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Parameters
        ----------
        cache_dir : Path, optional
            缓存目录，默认使用配置文件中的RAW_DATA_DIR
        """
        self.cache_dir = cache_dir or RAW_DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CacheManager initialized with cache_dir: {self.cache_dir}")

    def _generate_cache_key(
        self,
        market: str,
        universe: str,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> str:
        """
        生成缓存键

        Parameters
        ----------
        market : str
            市场类型
        universe : str
            股票池名称
        tickers : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期

        Returns
        -------
        str
            缓存文件名
        """
        # 对tickers排序以确保一致性
        tickers_sorted = sorted(tickers)
        tickers_str = "_".join(tickers_sorted[:5])  # 只用前5个ticker避免文件名过长

        # 生成hash
        hash_input = f"{market}_{universe}_{len(tickers)}_{start_date}_{end_date}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        cache_key = f"{market}_{universe}_{start_date}_{end_date}_{len(tickers)}tickers_{hash_suffix}"
        return cache_key

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.parquet"

    def _is_cache_valid(self, cache_path: Path, expiry_days: int = CACHE_EXPIRY_DAYS) -> bool:
        """
        检查缓存是否有效

        Parameters
        ----------
        cache_path : Path
            缓存文件路径
        expiry_days : int
            过期天数

        Returns
        -------
        bool
            缓存是否有效
        """
        if not cache_path.exists():
            return False

        # 检查文件修改时间
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime

        if age > timedelta(days=expiry_days):
            logger.info(f"Cache expired (age: {age.days} days): {cache_path.name}")
            return False

        return True

    def save(self, data: pd.DataFrame, cache_key: str) -> Path:
        """
        保存数据到缓存

        Parameters
        ----------
        data : pd.DataFrame
            要缓存的数据
        cache_key : str
            缓存键

        Returns
        -------
        Path
            缓存文件路径
        """
        cache_path = self._get_cache_path(cache_key)

        try:
            data.to_parquet(
                cache_path,
                compression=CACHE_COMPRESSION,
                index=True
            )
            file_size = cache_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Saved cache: {cache_path.name} ({file_size:.2f} MB)")
            return cache_path

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            raise

    def load(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据

        Parameters
        ----------
        cache_key : str
            缓存键

        Returns
        -------
        pd.DataFrame or None
            缓存的数据，如果不存在或过期返回None
        """
        cache_path = self._get_cache_path(cache_key)

        if not self._is_cache_valid(cache_path):
            return None

        try:
            data = pd.read_parquet(cache_path)
            logger.info(f"Loaded cache: {cache_path.name} (shape: {data.shape})")
            return data

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    def get_or_fetch(
        self,
        market: str,
        universe: str,
        tickers: List[str],
        start_date: str,
        end_date: str,
        fetcher,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取缓存数据，如果不存在则从API获取

        Parameters
        ----------
        market : str
            市场类型
        universe : str
            股票池名称
        tickers : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        fetcher : YFinanceFetcher
            数据获取器实例
        force_refresh : bool
            是否强制刷新缓存

        Returns
        -------
        pd.DataFrame
            股票数据
        """
        cache_key = self._generate_cache_key(market, universe, tickers, start_date, end_date)

        # 尝试从缓存加载
        if not force_refresh:
            cached_data = self.load(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached data for {market} {universe}")
                return cached_data

        # 从API获取
        logger.info(f"Fetching fresh data for {market} {universe}")
        data = fetcher.fetch_in_batches(tickers, start_date, end_date)

        if not data.empty:
            # 保存到缓存
            self.save(data, cache_key)

        return data

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        清理缓存

        Parameters
        ----------
        older_than_days : int, optional
            清理多少天前的缓存，None表示清理所有
        """
        logger.info(f"Clearing cache in {self.cache_dir}")

        cleared_count = 0
        cleared_size = 0

        for cache_file in self.cache_dir.glob("*.parquet"):
            should_delete = False

            if older_than_days is None:
                should_delete = True
            else:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                age = datetime.now() - mtime
                if age > timedelta(days=older_than_days):
                    should_delete = True

            if should_delete:
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                cleared_count += 1
                cleared_size += file_size
                logger.debug(f"Deleted: {cache_file.name}")

        cleared_size_mb = cleared_size / (1024 * 1024)
        logger.info(f"Cleared {cleared_count} files ({cleared_size_mb:.2f} MB)")

    def get_cache_info(self) -> dict:
        """
        获取缓存信息

        Returns
        -------
        dict
            缓存统计信息
        """
        cache_files = list(self.cache_dir.glob("*.parquet"))

        total_size = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size / (1024 * 1024)

        if cache_files:
            oldest = min(cache_files, key=lambda f: f.stat().st_mtime)
            newest = max(cache_files, key=lambda f: f.stat().st_mtime)
            oldest_age = datetime.now() - datetime.fromtimestamp(oldest.stat().st_mtime)
            newest_age = datetime.now() - datetime.fromtimestamp(newest.stat().st_mtime)
        else:
            oldest_age = newest_age = None

        return {
            'cache_dir': str(self.cache_dir),
            'file_count': len(cache_files),
            'total_size_mb': total_size_mb,
            'oldest_file_age_days': oldest_age.days if oldest_age else None,
            'newest_file_age_days': newest_age.days if newest_age else None,
        }


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher

    # 配置日志
    logger.add("cache_manager.log", rotation="10 MB")

    # 初始化
    cache_manager = CacheManager()
    fetcher = YFinanceFetcher()

    # 测试缓存
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    market = "US"
    universe = "test"
    start = "2023-01-01"
    end = "2024-01-01"

    print("\n=== First fetch (from API) ===")
    data1 = cache_manager.get_or_fetch(market, universe, test_tickers, start, end, fetcher)
    print(f"Data shape: {data1.shape}")

    print("\n=== Second fetch (from cache) ===")
    data2 = cache_manager.get_or_fetch(market, universe, test_tickers, start, end, fetcher)
    print(f"Data shape: {data2.shape}")

    print("\n=== Cache Info ===")
    info = cache_manager.get_cache_info()
    for key, value in info.items():
        print(f"{key}: {value}")
