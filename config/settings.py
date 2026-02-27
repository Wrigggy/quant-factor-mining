"""
全局配置文件
定义数据路径、API限制、因子参数、优化参数等
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

# 确保目录存在
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FACTORS_DATA_DIR, ALPHALENS_DIR, BACKTESTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== 数据获取配置 ====================
# YFinance API限制
YFINANCE_RATE_LIMIT = 2000  # 每小时请求数（保守估计）
YFINANCE_BATCH_SIZE = 50    # 每批下载股票数
YFINANCE_MAX_WORKERS = 5    # 并行下载线程数
YFINANCE_RETRY_TIMES = 3    # 失败重试次数
YFINANCE_RETRY_DELAY = 5    # 重试间隔（秒）

# 数据时间范围
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# 缓存配置
CACHE_EXPIRY_DAYS = 7  # 缓存过期时间（天）
CACHE_FORMAT = "parquet"  # 缓存格式
CACHE_COMPRESSION = "snappy"  # 压缩算法

# ==================== 数据质量配置 ====================
# 数据验证阈值
MAX_MISSING_RATIO = 0.5  # 最大缺失值比例（50%）
MAX_CONSECUTIVE_MISSING = 5  # 最大连续缺失天数
MAX_SINGLE_DAY_RETURN = 1.0  # 最大单日收益率（100%）

# Winsorize参数
WINSORIZE_LOWER = 0.01  # 1%分位数
WINSORIZE_UPPER = 0.99  # 99%分位数

# ==================== 因子计算配置 ====================
# 动量因子参数
MOMENTUM_LOOKBACK = 252  # 12个月回看期（交易日）
MOMENTUM_SKIP = 21       # 跳过最后1个月

# 均值回归因子参数
MEAN_REVERSION_LOOKBACK = 21  # 1个月回看期

# 波动率因子参数
VOLATILITY_WINDOW = 63  # 3个月滚动窗口
VOLATILITY_ANNUALIZE = True  # 是否年化

# 因子标准化方法
FACTOR_NORMALIZATION_METHOD = "zscore"  # 可选: "zscore", "rank"

# ==================== Alphalens评估配置 ====================
# 前瞻收益率周期（交易日）
ALPHALENS_PERIODS = [1, 5, 21]  # 1天、5天、21天

# 分位数数量
ALPHALENS_QUANTILES = 5

# IC显著性阈值
IC_SIGNIFICANCE_THRESHOLD = 0.02  # IC均值阈值
IC_TSTAT_THRESHOLD = 2.0          # t统计量阈值
IC_CONSISTENCY_THRESHOLD = 0.6    # IC一致性阈值（正值月份占比）

# ==================== 组合优化配置 ====================
# 优化参数
RISK_AVERSION = 1.0  # 风险厌恶系数
MAX_POSITION_SIZE = 0.05  # 单只股票最大持仓比例（5%）
TURNOVER_LIMIT = 0.5  # 月度换手率上限（50%）
LONG_ONLY = True  # 仅多头策略

# 协方差矩阵估计
COV_ESTIMATION_METHOD = "ledoit_wolf"  # 可选: "sample", "ledoit_wolf", "factor_model"
COV_ESTIMATION_WINDOW = 252  # 协方差估计窗口（1年）

# ==================== 回测配置 ====================
# 回测参数
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
