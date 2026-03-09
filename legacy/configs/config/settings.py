"""
Global Configuration File
Defines data paths, API limits, factor parameters, optimization parameters, etc.
"""

import os
from pathlib import Path
from datetime import datetime

# ==================== 项目路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FACTORS_DATA_DIR = DATA_DIR / "factors"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ALPHALENS_DIR = OUTPUTS_DIR / "alphalens"
BACKTESTS_DIR = OUTPUTS_DIR / "backtests"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FACTORS_DATA_DIR, ALPHALENS_DIR, BACKTESTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== 数据获取配置 ====================
# YFinance API limits
YFINANCE_RATE_LIMIT = 2000  # Requests per hour (conservative estimate)
YFINANCE_BATCH_SIZE = 50    # Stocks per batch download
YFINANCE_MAX_WORKERS = 5    # Parallel download threads
YFINANCE_RETRY_TIMES = 3    # Retry attempts on failure
YFINANCE_RETRY_DELAY = 5    # Retry interval (seconds)

# Data time range
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# Cache configuration
CACHE_EXPIRY_DAYS = 7  # Cache expiry time (days)
CACHE_FORMAT = "parquet"  # Cache format
CACHE_COMPRESSION = "snappy"  # Compression algorithm

# ==================== 数据质量配置 ====================
# Data validation thresholds
MAX_MISSING_RATIO = 0.5  # Maximum missing value ratio (50%)
MAX_CONSECUTIVE_MISSING = 5  # Maximum consecutive missing days
MAX_SINGLE_DAY_RETURN = 1.0  # Maximum single day return (100%)

# Winsorize parameters
WINSORIZE_LOWER = 0.01  # 1% quantile
WINSORIZE_UPPER = 0.99  # 99% quantile

# ==================== 因子计算配置 ====================
# Momentum factor parameters
MOMENTUM_LOOKBACK = 252  # 12-month lookback period (trading days)
MOMENTUM_SKIP = 21       # Skip last 1 month

# Mean reversion factor parameters
MEAN_REVERSION_LOOKBACK = 21  # 1-month lookback period

# Volatility factor parameters
VOLATILITY_WINDOW = 63  # 3-month rolling window
VOLATILITY_ANNUALIZE = True  # Whether to annualize

# Factor normalization method
FACTOR_NORMALIZATION_METHOD = "zscore"  # Options: "zscore", "rank"

# ==================== Alphalens评估配置 ====================
# Forward return periods (trading days)
ALPHALENS_PERIODS = [1, 5, 21]  # 1 day, 5 days, 21 days

# Number of quantiles
ALPHALENS_QUANTILES = 5

# IC significance thresholds
IC_SIGNIFICANCE_THRESHOLD = 0.02  # IC mean threshold
IC_TSTAT_THRESHOLD = 2.0          # t-statistic threshold
IC_CONSISTENCY_THRESHOLD = 0.6    # IC consistency threshold (positive month ratio)

# ==================== 组合优化配置 ====================
# Optimization parameters
RISK_AVERSION = 1.0  # Risk aversion coefficient
MAX_POSITION_SIZE = 0.05  # Maximum position size per stock (5%)
TURNOVER_LIMIT = 0.5  # Monthly turnover limit (50%)
LONG_ONLY = True  # Long-only strategy

# Covariance matrix estimation
COV_ESTIMATION_METHOD = "ledoit_wolf"  # Options: "sample", "ledoit_wolf", "factor_model"
COV_ESTIMATION_WINDOW = 252  # Covariance estimation window (1 year)

# ==================== 回测配置 ====================
# Backtest parameters
INITIAL_CAPITAL = 1_000_000  # 初始资金（$1M）
TRANSACTION_COST = 0.001     # 交易成本（10 bps 单边）
REBALANCE_FREQUENCY = 21     # 调仓频率（月度，21个交易日）
RISK_FREE_RATE = 0.02        # 无风险利率（年化）

# ==================== 日志配置 ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# ==================== 数据库配置（可选，未来扩展） ====================
# 如果需要使用数据库存储数据，可以在这里配置
DATABASE_URL = os.getenv("DATABASE_URL", None)
