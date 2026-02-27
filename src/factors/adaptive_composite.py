"""
自适应因子组合模块

核心功能：
1. AdaptiveCompositeFactor - 基于滚动IC动态调整因子权重
2. RegimeDetector - 市场状态检测（趋势/均值回归/高波动）
3. RegimeAwareCompositeFactor - 结合市场状态的自适应因子组合

设计原则：
- 避免前视偏差：IC计算延迟forward_period天
- 指数衰减：近期IC权重更高
- 最小权重约束：防止因子权重为零
- 向后兼容：继承自BaseFactor和CompositeFactor
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from loguru import logger

from .base import BaseFactor, CompositeFactor


class AdaptiveCompositeFactor(BaseFactor):
    """
    基于滚动IC的自适应因子组合

    核心思想：
    - 计算每个因子的滚动IC（信息系数）
    - IC高的因子获得更高权重
    - 使用指数衰减：近期IC比历史IC更重要

    Parameters
    ----------
    factors : List[BaseFactor]
        基础因子列表
    ic_window : int, default=63
        滚动IC窗口长度（交易日数）
    decay_halflife : int, default=21
        指数衰减半衰期（天）
    min_weight : float, default=0.05
        单因子最小权重（防止完全归零）
    normalization_method : str, default='softmax'
        权重标准化方法：'softmax', 'relu', 'rank'
    """

    def __init__(
        self,
        factors: List[BaseFactor],
        ic_window: int = 63,
        decay_halflife: int = 21,
        min_weight: float = 0.05,
        normalization_method: str = 'softmax',
        name: str = "AdaptiveComposite"
    ):
        super().__init__(name)
        self.factors = factors
        self.ic_window = ic_window
        self.decay_halflife = decay_halflife
        self.min_weight = min_weight
        self.normalization_method = normalization_method

        # IC历史存储：{factor_name: [(date, ic_value), ...]}
        self.ic_history: Dict[str, List[tuple]] = {
            f.name: [] for f in factors
        }

        # 权重历史：用于分析和调试
        self.weight_history: List[Dict] = []

        logger.info(f"AdaptiveCompositeFactor初始化：{len(factors)}个因子")
        logger.info(f"  IC窗口={ic_window}天，衰减半衰期={decay_halflife}天")
        logger.info(f"  最小权重={min_weight}，标准化方法={normalization_method}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算自适应组合因子

        注意：此方法不包含IC更新逻辑（需要未来收益率）
        实际使用时应通过AdaptiveCompositeManager管理

        Parameters
        ----------
        data : pd.DataFrame
            多股票多日期的OHLCV数据
            Index: MultiIndex(date, ticker)
            Columns: [open, high, low, close, volume]

        Returns
        -------
        pd.Series
            自适应组合因子值
        """
        current_date = data.index.get_level_values('date').max()

        # 1. 计算所有基础因子
        factor_values = {}
        for factor in self.factors:
            logger.debug(f"计算因子：{factor.name}")
            values = factor.calculate(data)
            # 标准化到[-1, 1]范围
            values = self._normalize_factor(values)
            factor_values[factor.name] = values

        # 2. 计算自适应权重（基于历史IC）
        weights = self._compute_adaptive_weights(current_date)

        # 3. 加权组合
        composite = pd.Series(0.0, index=factor_values[self.factors[0].name].index)
        for factor in self.factors:
            composite += factor_values[factor.name] * weights[factor.name]

        # 4. 记录权重历史
        self.weight_history.append({
            'date': current_date,
            'weights': weights.copy()
        })

        logger.debug(f"日期{current_date}的因子权重：{weights}")

        return composite

    def _normalize_factor(self, factor_values: pd.Series) -> pd.Series:
        """标准化单个因子到[-1, 1]范围"""
        # 截面标准化（每个日期独立）
        normalized = factor_values.groupby(level='date').apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )
        # Clip到[-3, 3]避免极端值
        normalized = normalized.clip(-3, 3)
        return normalized

    def _compute_adaptive_weights(self, current_date) -> Dict[str, float]:
        """
        基于滚动IC计算自适应权重

        算法：
        1. 取每个因子最近ic_window个IC值
        2. 应用指数衰减权重（近期IC更重要）
        3. 计算加权平均IC
        4. 转换为正权重（softmax/relu/rank）
        5. 标准化使权重和为1
        """
        weights = {}

        for factor in self.factors:
            ic_list = self.ic_history[factor.name]

            # 预热期：IC历史不足时使用等权重
            if len(ic_list) < self.ic_window:
                weights[factor.name] = 1.0 / len(self.factors)
                continue

            # 取最近ic_window个IC值
            recent_ic = ic_list[-self.ic_window:]
            dates = [x[0] for x in recent_ic]
            ic_values = np.array([x[1] for x in recent_ic])

            # 应用指数衰减：decay_weight = exp(-ln(2) * age / halflife)
            # age从0（最新）到ic_window-1（最老）
            ages = np.arange(len(ic_values))[::-1]  # [ic_window-1, ..., 1, 0]
            decay_weights = np.exp(-np.log(2) * ages / self.decay_halflife)

            # 加权平均IC
            weighted_ic = np.average(ic_values, weights=decay_weights)
            weights[factor.name] = weighted_ic

        # 转换为正权重
        weights = self._normalize_weights(weights)

        return weights

    def _normalize_weights(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """
        将原始IC分数转换为正权重（和为1）

        支持三种方法：
        - softmax: exp(IC) / sum(exp(IC))
        - relu: max(IC+bias, 0) / sum(max(IC+bias, 0))
        - rank: rank(IC) / sum(ranks)
        """
        if self.normalization_method == 'softmax':
            # Softmax with temperature scaling
            ic_values = np.array(list(raw_weights.values()))
            # 温度参数：避免权重过于极端
            temperature = 2.0
            exp_ic = np.exp(ic_values / temperature)
            softmax_weights = exp_ic / exp_ic.sum()

            weights = dict(zip(raw_weights.keys(), softmax_weights))

        elif self.normalization_method == 'relu':
            # ReLU + minimum weight
            positive_weights = {k: max(v, 0) + self.min_weight
                              for k, v in raw_weights.items()}
            total = sum(positive_weights.values())
            weights = {k: v / total for k, v in positive_weights.items()}

        elif self.normalization_method == 'rank':
            # Rank-based weights
            sorted_items = sorted(raw_weights.items(), key=lambda x: x[1])
            ranks = {name: i+1 for i, (name, _) in enumerate(sorted_items)}
            total_rank = sum(ranks.values())
            weights = {k: v / total_rank for k, v in ranks.items()}

        else:
            raise ValueError(f"未知的标准化方法：{self.normalization_method}")

        return weights

    def update_ic_history(
        self,
        factor_name: str,
        date,
        ic_value: float
    ):
        """
        更新因子的IC历史

        注意：此方法应在计算出forward returns后调用
        通常由AdaptiveCompositeManager管理

        Parameters
        ----------
        factor_name : str
            因子名称
        date : pd.Timestamp or str
            日期
        ic_value : float
            IC值（Spearman相关系数）
        """
        if factor_name not in self.ic_history:
            logger.warning(f"未知因子：{factor_name}")
            return

        self.ic_history[factor_name].append((date, ic_value))
        logger.debug(f"更新{factor_name}的IC历史：日期={date}, IC={ic_value:.4f}")

    def get_weight_history(self) -> pd.DataFrame:
        """
        获取权重演化历史

        Returns
        -------
        pd.DataFrame
            Index: date
            Columns: [factor1, factor2, ...]
        """
        if not self.weight_history:
            return pd.DataFrame()

        records = []
        for entry in self.weight_history:
            record = {'date': entry['date']}
            record.update(entry['weights'])
            records.append(record)

        df = pd.DataFrame(records).set_index('date')
        return df

    def get_ic_summary(self) -> pd.DataFrame:
        """
        获取各因子IC统计摘要

        Returns
        -------
        pd.DataFrame
            Index: factor_name
            Columns: [ic_mean, ic_std, ic_ir, n_observations]
        """
        summary = []

        for factor_name, ic_list in self.ic_history.items():
            if len(ic_list) == 0:
                continue

            ic_values = np.array([x[1] for x in ic_list])

            summary.append({
                'factor_name': factor_name,
                'ic_mean': ic_values.mean(),
                'ic_std': ic_values.std(),
                'ic_ir': ic_values.mean() / (ic_values.std() + 1e-9),
                'n_observations': len(ic_values),
                'ic_positive_rate': (ic_values > 0).mean()
            })

        df = pd.DataFrame(summary).set_index('factor_name')
        return df


class AdaptiveCompositeManager:
    """
    自适应因子组合管理器

    职责：
    1. 管理因子计算的时间顺序
    2. 避免前视偏差（lookahead bias）
    3. 延迟更新IC历史（等待forward returns可用）

    工作流程：
    - t时刻：计算因子值（使用t-forward_period之前的IC历史确定权重）
    - t+forward_period时刻：计算t时刻的IC，更新IC历史
    """

    def __init__(
        self,
        adaptive_factor: AdaptiveCompositeFactor,
        forward_period: int = 21
    ):
        """
        Parameters
        ----------
        adaptive_factor : AdaptiveCompositeFactor
            自适应因子实例
        forward_period : int, default=21
            前瞻收益率周期（交易日数）
        """
        self.adaptive_factor = adaptive_factor
        self.forward_period = forward_period

        # 缓冲区：存储因子值和价格，用于延迟IC计算
        self.factor_buffer: List[tuple] = []  # (date, factor_values_dict)
        self.price_buffer: List[tuple] = []   # (date, prices)

        logger.info(f"AdaptiveCompositeManager初始化：forward_period={forward_period}天")

    def compute_factor_with_update(
        self,
        data: pd.DataFrame,
        prices: pd.Series,
        current_date
    ) -> tuple:
        """
        计算因子并更新IC历史（无前视偏差）

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV数据
        prices : pd.Series
            收盘价（用于计算收益率）
            Index: MultiIndex(date, ticker)
        current_date : pd.Timestamp
            当前日期

        Returns
        -------
        composite_factor : pd.Series
            组合因子值
        current_weights : Dict[str, float]
            当前权重
        """
        # 1. 计算所有基础因子的值
        base_factor_values = {}
        for factor in self.adaptive_factor.factors:
            values = factor.calculate(data)
            base_factor_values[factor.name] = values

        # 2. 存储到缓冲区
        self.factor_buffer.append((current_date, base_factor_values))
        self.price_buffer.append((current_date, prices))

        # 3. 检查是否可以计算forward_period天前的IC
        if len(self.factor_buffer) > self.forward_period:
            past_date, past_factors = self.factor_buffer[-self.forward_period-1]

            # 计算从past_date到current_date的收益率
            past_prices = self.price_buffer[-self.forward_period-1][1]
            current_prices = prices

            # Forward returns
            forward_returns = (current_prices / past_prices - 1)

            # 计算每个因子的IC
            for factor_name, factor_values in past_factors.items():
                # 只取past_date的因子值
                factor_at_past = factor_values[
                    factor_values.index.get_level_values('date') == past_date
                ]

                # 对齐
                aligned = pd.DataFrame({
                    'factor': factor_at_past.values,
                    'returns': forward_returns.reindex(
                        factor_at_past.index.get_level_values('ticker')
                    ).values
                }).dropna()

                if len(aligned) < 10:
                    continue

                # Spearman IC
                from scipy.stats import spearmanr
                ic_value, _ = spearmanr(aligned['factor'], aligned['returns'])

                # 更新IC历史
                self.adaptive_factor.update_ic_history(
                    factor_name, past_date, ic_value
                )

        # 4. 使用当前权重计算组合因子
        composite_factor = self.adaptive_factor.calculate(data)
        current_weights = self.adaptive_factor.weight_history[-1]['weights']

        return composite_factor, current_weights


class RegimeDetector:
    """
    市场状态检测器

    检测三种市场状态：
    1. Trending（趋势市）：正自相关，适合动量策略
    2. Mean-Reverting（均值回归市）：负自相关，适合反转策略
    3. High-Volatility（高波动市）：波动率高，降低风险敞口
    """

    def __init__(self, lookback: int = 63):
        """
        Parameters
        ----------
        lookback : int, default=63
            回看窗口（交易日数）
        """
        self.lookback = lookback
        logger.info(f"RegimeDetector初始化：lookback={lookback}天")

    def detect_regime(
        self,
        market_returns: pd.Series,
        volume: Optional[pd.Series] = None
    ) -> str:
        """
        检测当前市场状态

        Parameters
        ----------
        market_returns : pd.Series
            市场收益率时间序列（如大盘指数收益率）
        volume : pd.Series, optional
            成交量（暂未使用）

        Returns
        -------
        str
            'trending', 'mean_reverting', 'high_vol', 'neutral'
        """
        if len(market_returns) < self.lookback:
            return 'neutral'

        recent = market_returns.tail(self.lookback)

        # 1. 自相关性（lag-1）
        autocorr = recent.autocorr(lag=1)

        # 2. 波动率
        vol = recent.std() * np.sqrt(252)  # 年化

        # 历史波动率分布
        rolling_vol = market_returns.rolling(self.lookback).std() * np.sqrt(252)
        vol_percentile = (rolling_vol < vol).mean()

        # 3. 判断状态
        if vol_percentile > 0.8:
            # 当前波动率超过历史80%分位数
            return 'high_vol'
        elif autocorr > 0.15:
            # 强正自相关
            return 'trending'
        elif autocorr < -0.15:
            # 强负自相关
            return 'mean_reverting'
        else:
            return 'neutral'


class RegimeAwareCompositeFactor(AdaptiveCompositeFactor):
    """
    结合市场状态的自适应因子组合

    扩展AdaptiveCompositeFactor，根据市场状态调整因子权重：
    - Trending市：增加动量因子权重
    - Mean-Reverting市：增加反转因子权重
    - High-Vol市：降低所有因子权重（或增加低波因子）
    """

    def __init__(
        self,
        factors: List[BaseFactor],
        regime_detector: RegimeDetector,
        market_returns: pd.Series,
        regime_boost_factor: float = 1.5,
        **kwargs
    ):
        """
        Parameters
        ----------
        factors : List[BaseFactor]
            基础因子列表
        regime_detector : RegimeDetector
            市场状态检测器
        market_returns : pd.Series
            市场收益率时间序列（用于状态检测）
        regime_boost_factor : float, default=1.5
            状态匹配时的权重提升倍数
        **kwargs : dict
            传递给AdaptiveCompositeFactor的其他参数
        """
        super().__init__(factors, **kwargs)
        self.regime_detector = regime_detector
        self.market_returns = market_returns
        self.regime_boost_factor = regime_boost_factor

        # 因子类别映射（用于状态匹配）
        self.factor_categories = self._categorize_factors()

        # 状态历史
        self.regime_history: List[tuple] = []  # (date, regime)

        logger.info("RegimeAwareCompositeFactor初始化")
        logger.info(f"  因子分类：{self.factor_categories}")

    def _categorize_factors(self) -> Dict[str, str]:
        """
        将因子分类为动量/反转/波动率

        基于因子名称的启发式分类
        """
        categories = {}

        for factor in self.factors:
            name_lower = factor.name.lower()

            if 'momentum' in name_lower or 'trend' in name_lower:
                categories[factor.name] = 'momentum'
            elif 'reversion' in name_lower or 'reversal' in name_lower:
                categories[factor.name] = 'mean_reversion'
            elif 'volatility' in name_lower or 'vol' in name_lower:
                categories[factor.name] = 'volatility'
            else:
                categories[factor.name] = 'other'

        return categories

    def _compute_adaptive_weights(self, current_date) -> Dict[str, float]:
        """
        重写权重计算，加入市场状态调整
        """
        # 1. 获取基础IC权重
        base_weights = super()._compute_adaptive_weights(current_date)

        # 2. 检测当前市场状态
        regime = self.regime_detector.detect_regime(self.market_returns)
        self.regime_history.append((current_date, regime))

        logger.info(f"日期{current_date}市场状态：{regime}")

        # 3. 根据状态调整权重
        adjusted_weights = base_weights.copy()

        if regime == 'trending':
            # 提升动量因子权重
            for factor_name, category in self.factor_categories.items():
                if category == 'momentum':
                    adjusted_weights[factor_name] *= self.regime_boost_factor
                    logger.debug(f"提升{factor_name}权重（趋势市）")

        elif regime == 'mean_reverting':
            # 提升反转因子权重
            for factor_name, category in self.factor_categories.items():
                if category == 'mean_reversion':
                    adjusted_weights[factor_name] *= self.regime_boost_factor
                    logger.debug(f"提升{factor_name}权重（均值回归市）")

        elif regime == 'high_vol':
            # 提升低波因子权重，降低其他因子
            for factor_name, category in self.factor_categories.items():
                if category == 'volatility':
                    adjusted_weights[factor_name] *= self.regime_boost_factor
                else:
                    adjusted_weights[factor_name] *= 0.7
            logger.debug("高波动市场：提升低波因子，降低其他因子")

        # 4. 重新标准化
        adjusted_weights = self._normalize_weights(adjusted_weights)

        return adjusted_weights

    def get_regime_history(self) -> pd.DataFrame:
        """
        获取市场状态历史

        Returns
        -------
        pd.DataFrame
            Index: date
            Columns: [regime]
        """
        if not self.regime_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.regime_history, columns=['date', 'regime'])
        df = df.set_index('date')
        return df
