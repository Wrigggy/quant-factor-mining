"""
股票池定义
包含美股（S&P 500）和港股主板股票列表
"""

from typing import List, Dict
import pandas as pd

# ==================== 美股股票池 ====================

# S&P 500主要成分股（示例，实际应使用完整列表或动态获取）
# 这里列出部分代表性股票，实际使用时应获取完整列表
SP500_SAMPLE = [
    # 科技
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "ADBE",
    "CRM", "CSCO", "INTC", "AMD", "QCOM", "TXN", "INTU", "AMAT", "MU", "LRCX",

    # 金融
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "SPGI",
    "CME", "ICE", "CB", "PGR", "TFC", "USB", "PNC", "COF", "BK", "STT",

    # 医疗
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "CI", "ELV", "MCK", "COR", "VRTX", "REGN", "ISRG",

    # 消费
    "WMT", "HD", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT",
    "LOW", "TJX", "EL", "DG", "ROST", "ORLY", "AZO", "CMG", "YUM", "ULTA",

    # 工业
    "BA", "CAT", "GE", "HON", "UNP", "RTX", "LMT", "UPS", "DE", "MMM",
    "GD", "NOC", "EMR", "ETN", "ITW", "WM", "CSX", "NSC", "FDX", "PCAR",

    # 能源
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HES",

    # 材料
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DD", "DOW", "NUE", "VMC",

    # 房地产
    "AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "WELL", "AVB", "EQR",

    # 公用事业
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ES",

    # 通信
    "TMUS", "VZ", "T", "DIS", "CMCSA", "NFLX", "CHTR", "EA", "TTWO", "LYV",
]

# ==================== 港股股票池 ====================

# 恒生指数主要成分股（示例）
# 注意：港股代码格式为 "XXXX.HK"
HSI_SAMPLE = [
    # 科技
    "0700.HK",  # 腾讯控股
    "9988.HK",  # 阿里巴巴-SW
    "9618.HK",  # 京东集团-SW
    "1024.HK",  # 快手-W
    "9999.HK",  # 网易-S
    "0981.HK",  # 中芯国际
    "2382.HK",  # 舜宇光学科技
    "1810.HK",  # 小米集团-W
    "9626.HK",  # 哔哩哔哩-W
    "3690.HK",  # 美团-W

    # 金融
    "0005.HK",  # 汇丰控股
    "1398.HK",  # 工商银行
    "3988.HK",  # 中国银行
    "0939.HK",  # 建设银行
    "0388.HK",  # 香港交易所
    "1299.HK",  # 友邦保险
    "2318.HK",  # 中国平安
    "2628.HK",  # 中国人寿
    "1288.HK",  # 农业银行
    "3968.HK",  # 招商银行

    # 地产
    "0016.HK",  # 新鸿基地产
    "1997.HK",  # 九龙仓集团
    "0012.HK",  # 恒基地产
    "0688.HK",  # 中国海外发展
    "1109.HK",  # 华润置地
    "0101.HK",  # 恒隆地产
    "1113.HK",  # 长实集团
    "0017.HK",  # 新世界发展

    # 能源
    "0857.HK",  # 中国石油股份
    "0386.HK",  # 中国石油化工股份
    "0883.HK",  # 中国海洋石油
    "2688.HK",  # 新奥能源
    "3988.HK",  # 中国中煤能源

    # 消费
    "1876.HK",  # 百威亚太
    "2319.HK",  # 蒙牛乳业
    "0291.HK",  # 华润啤酒
    "6098.HK",  # 碧桂园服务
    "1928.HK",  # 金沙中国
    "0027.HK",  # 银河娱乐
    "0880.HK",  # 澳博控股

    # 工业
    "0002.HK",  # 中电控股
    "0003.HK",  # 香港中华煤气
    "1038.HK",  # 长江基建
    "0006.HK",  # 电能实业
    "1972.HK",  # 太古地产
    "1113.HK",  # 长实集团

    # 医药
    "1177.HK",  # 中国生物制药
    "1093.HK",  # 石药集团
    "2269.HK",  # 药明生物
    "6185.HK",  # 康希诺生物-B
    "9995.HK",  # 荣昌生物-B
]

# ==================== 股票池获取函数 ====================

def get_universe(market: str, universe_type: str = "full") -> List[str]:
    """
    获取指定市场的股票池

    Parameters
    ----------
    market : str
        市场类型，可选 "US" 或 "HK"
    universe_type : str
        股票池类型，可选 "full" 或 "sample"

    Returns
    -------
    List[str]
        股票代码列表

    Examples
    --------
    >>> get_universe("US", "sample")
    ['AAPL', 'MSFT', 'GOOGL', ...]

    >>> get_universe("HK", "sample")
    ['0700.HK', '9988.HK', ...]
    """
    if market.upper() == "US":
        if universe_type == "sample":
            return SP500_SAMPLE
        elif universe_type == "full":
            # 实际应用中，应从标准数据源获取完整S&P 500列表
            # 例如：从Wikipedia或使用专门的库
            print("Warning: Using sample S&P 500 list. For production, fetch complete list.")
            return SP500_SAMPLE
    elif market.upper() == "HK":
        if universe_type == "sample":
            return HSI_SAMPLE
        elif universe_type == "full":
            print("Warning: Using sample HSI list. For production, fetch complete list.")
            return HSI_SAMPLE
    else:
        raise ValueError(f"Unknown market: {market}. Must be 'US' or 'HK'")


def get_sp500_full() -> List[str]:
    """
    从Wikipedia动态获取完整S&P 500列表

    Returns
    -------
    List[str]
        S&P 500股票代码列表
    """
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        # 修正某些特殊字符
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        print(f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        print("Falling back to sample list")
        return SP500_SAMPLE


def get_universe_info() -> Dict[str, Dict]:
    """
    获取所有股票池的信息

    Returns
    -------
    Dict[str, Dict]
        股票池信息字典
    """
    return {
        "US": {
            "name": "S&P 500",
            "sample_size": len(SP500_SAMPLE),
            "description": "美国标普500指数成分股"
        },
        "HK": {
            "name": "HSI Components",
            "sample_size": len(HSI_SAMPLE),
            "description": "香港恒生指数成分股"
        }
    }


if __name__ == "__main__":
    # 测试代码
    print("=== Universe Information ===")
    info = get_universe_info()
    for market, details in info.items():
        print(f"\n{market}: {details['name']}")
        print(f"  Sample size: {details['sample_size']}")
        print(f"  Description: {details['description']}")

    print("\n=== US Sample Tickers (first 10) ===")
    us_tickers = get_universe("US", "sample")
    print(us_tickers[:10])

    print("\n=== HK Sample Tickers (first 10) ===")
    hk_tickers = get_universe("HK", "sample")
    print(hk_tickers[:10])
