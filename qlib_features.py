"""
Qlib 风格的特征工程模块
实现类似 Qlib 的 Alpha 特征表达式
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from abc import ABC, abstractmethod


class FeatureExpression(ABC):
    """特征表达式基类（类似 Qlib 的 Expression）"""
    
    @abstractmethod
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        """计算特征值"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取特征名称"""
        pass


class FieldFeature(FeatureExpression):
    """字段特征（原始数据字段）"""
    
    def __init__(self, field_name: str):
        self.field_name = field_name
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        if self.field_name in feature_dict:
            return feature_dict[self.field_name]
        return df[self.field_name]
    
    def get_name(self) -> str:
        return self.field_name


class RefFeature(FeatureExpression):
    """引用过去 N 天的值（shift）"""
    
    def __init__(self, expr: FeatureExpression, n: int):
        self.expr = expr
        self.n = n
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        values = self.expr.evaluate(df, feature_dict)
        return values.groupby(df['code'], group_keys=False).shift(self.n)
    
    def get_name(self) -> str:
        return f"{self.expr.get_name()}_ref_{self.n}"


class MeanFeature(FeatureExpression):
    """移动平均（ts_mean）"""
    
    def __init__(self, expr: FeatureExpression, n: int):
        self.expr = expr
        self.n = n
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        values = self.expr.evaluate(df, feature_dict)
        return values.groupby(df['code'], group_keys=False).rolling(self.n, min_periods=1).mean().reset_index(0, drop=True)
    
    def get_name(self) -> str:
        return f"{self.expr.get_name()}_mean_{self.n}"


class StdFeature(FeatureExpression):
    """移动标准差（ts_std）"""
    
    def __init__(self, expr: FeatureExpression, n: int):
        self.expr = expr
        self.n = n
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        values = self.expr.evaluate(df, feature_dict)
        return values.groupby(df['code'], group_keys=False).rolling(self.n, min_periods=1).std().reset_index(0, drop=True)
    
    def get_name(self) -> str:
        return f"{self.expr.get_name()}_std_{self.n}"


class SumFeature(FeatureExpression):
    """移动求和（ts_sum）"""
    
    def __init__(self, expr: FeatureExpression, n: int):
        self.expr = expr
        self.n = n
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        values = self.expr.evaluate(df, feature_dict)
        return values.groupby(df['code'], group_keys=False).rolling(self.n, min_periods=1).sum().reset_index(0, drop=True)
    
    def get_name(self) -> str:
        return f"{self.expr.get_name()}_sum_{self.n}"


class MaxFeature(FeatureExpression):
    """移动最大值（ts_max）"""
    
    def __init__(self, expr: FeatureExpression, n: int):
        self.expr = expr
        self.n = n
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        values = self.expr.evaluate(df, feature_dict)
        return values.groupby(df['code'], group_keys=False).rolling(self.n, min_periods=1).max().reset_index(0, drop=True)
    
    def get_name(self) -> str:
        return f"{self.expr.get_name()}_max_{self.n}"


class MinFeature(FeatureExpression):
    """移动最小值（ts_min）"""
    
    def __init__(self, expr: FeatureExpression, n: int):
        self.expr = expr
        self.n = n
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        values = self.expr.evaluate(df, feature_dict)
        return values.groupby(df['code'], group_keys=False).rolling(self.n, min_periods=1).min().reset_index(0, drop=True)
    
    def get_name(self) -> str:
        return f"{self.expr.get_name()}_min_{self.n}"


class DeltaFeature(FeatureExpression):
    """差分（ts_delta）"""
    
    def __init__(self, expr: FeatureExpression, n: int):
        self.expr = expr
        self.n = n
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        values = self.expr.evaluate(df, feature_dict)
        return values.groupby(df['code'], group_keys=False).diff(self.n)
    
    def get_name(self) -> str:
        return f"{self.expr.get_name()}_delta_{self.n}"


class ReturnsFeature(FeatureExpression):
    """收益率（ts_returns）"""
    
    def __init__(self, expr: FeatureExpression, n: int):
        self.expr = expr
        self.n = n
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        values = self.expr.evaluate(df, feature_dict)
        prev = values.groupby(df['code'], group_keys=False).shift(self.n)
        return (values - prev) / prev
    
    def get_name(self) -> str:
        return f"{self.expr.get_name()}_returns_{self.n}"


class RankFeature(FeatureExpression):
    """截面排名（cs_rank）"""
    
    def __init__(self, expr: FeatureExpression):
        self.expr = expr
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        values = self.expr.evaluate(df, feature_dict)
        return values.groupby(df['date'], group_keys=False).rank(pct=True)
    
    def get_name(self) -> str:
        return f"{self.expr.get_name()}_rank"


class ScaleFeature(FeatureExpression):
    """截面标准化（cs_scale/zscore）"""
    
    def __init__(self, expr: FeatureExpression):
        self.expr = expr
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        values = self.expr.evaluate(df, feature_dict)
        mean = values.groupby(df['date'], group_keys=False).transform('mean')
        std = values.groupby(df['date'], group_keys=False).transform('std')
        return (values - mean) / (std + 1e-8)
    
    def get_name(self) -> str:
        return f"{self.expr.get_name()}_scale"


class BinaryOpFeature(FeatureExpression):
    """二元运算特征"""
    
    def __init__(self, left: FeatureExpression, right: FeatureExpression, op: str):
        self.left = left
        self.right = right
        self.op = op
    
    def evaluate(self, df: pd.DataFrame, feature_dict: Dict[str, pd.Series]) -> pd.Series:
        left_val = self.left.evaluate(df, feature_dict)
        right_val = self.right.evaluate(df, feature_dict)
        
        if self.op == '+':
            return left_val + right_val
        elif self.op == '-':
            return left_val - right_val
        elif self.op == '*':
            return left_val * right_val
        elif self.op == '/':
            return left_val / (right_val + 1e-8)
        elif self.op == '>':
            return (left_val > right_val).astype(int)
        elif self.op == '<':
            return (left_val < right_val).astype(int)
        else:
            raise ValueError(f"Unknown operator: {self.op}")
    
    def get_name(self) -> str:
        return f"({self.left.get_name()}_{self.op}_{self.right.get_name()})"


class QlibFeatureEngineer:
    """
    Qlib 风格的特征工程器
    使用表达式系统构建 Alpha 特征
    """
    
    def __init__(self):
        self.feature_exprs: List[FeatureExpression] = []
        self.feature_names: List[str] = []
        
    def add_feature(self, expr: FeatureExpression, name: Optional[str] = None):
        """添加特征表达式"""
        self.feature_exprs.append(expr)
        self.feature_names.append(name or expr.get_name())
        return self
    
    def build_alpha_features(self) -> 'QlibFeatureEngineer':
        """
        构建经典的 Alpha 特征（参考 WorldQuant 101）
        """
        # 价格字段
        close = FieldFeature('close')
        open_p = FieldFeature('open')
        high = FieldFeature('high')
        low = FieldFeature('low')
        volume = FieldFeature('volume')
        
        # === Alpha 1: 收益率相关 ===
        # 5日收益率
        self.add_feature(ReturnsFeature(close, 5), 'alpha_001')
        # 10日收益率
        self.add_feature(ReturnsFeature(close, 10), 'alpha_002')
        # 20日收益率
        self.add_feature(ReturnsFeature(close, 20), 'alpha_003')
        # 60日收益率
        self.add_feature(ReturnsFeature(close, 60), 'alpha_004')
        
        # === Alpha 2: 移动平均相关 ===
        # 价格/5日均线比率
        ma5 = MeanFeature(close, 5)
        self.add_feature(BinaryOpFeature(close, ma5, '/'), 'alpha_005')
        
        # 价格/20日均线比率
        ma20 = MeanFeature(close, 20)
        self.add_feature(BinaryOpFeature(close, ma20, '/'), 'alpha_006')
        
        # 5日/20日均线比率
        self.add_feature(BinaryOpFeature(ma5, ma20, '/'), 'alpha_007')
        
        # === Alpha 3: 波动率相关 ===
        # 20日价格波动率
        self.add_feature(StdFeature(close, 20), 'alpha_008')
        # 20日收益率波动率
        ret_1 = ReturnsFeature(close, 1)
        self.add_feature(StdFeature(ret_1, 20), 'alpha_009')
        
        # === Alpha 4: 成交量相关 ===
        # 5日平均成交量
        vol_ma5 = MeanFeature(volume, 5)
        # 20日平均成交量
        vol_ma20 = MeanFeature(volume, 20)
        # 量比（5日/20日成交量比）
        self.add_feature(BinaryOpFeature(vol_ma5, vol_ma20, '/'), 'alpha_010')
        
        # 成交量变化率
        self.add_feature(ReturnsFeature(volume, 5), 'alpha_011')
        
        # === Alpha 5: 价格区间相关 ===
        # 20日最高价
        high_20 = MaxFeature(high, 20)
        # 20日最低价
        low_20 = MinFeature(low, 20)
        # 价格在20日区间的位置
        pos_20 = BinaryOpFeature(
            BinaryOpFeature(close, low_20, '-'),
            BinaryOpFeature(high_20, low_20, '-'),
            '/'
        )
        self.add_feature(pos_20, 'alpha_012')
        
        # === Alpha 6: 动量相关 ===
        # 5日动量（价格变化）
        self.add_feature(DeltaFeature(close, 5), 'alpha_013')
        # 10日动量
        self.add_feature(DeltaFeature(close, 10), 'alpha_014')
        
        # === Alpha 7: 均值回归相关 ===
        # 价格偏离20日均线程度
        dev_20 = BinaryOpFeature(
            BinaryOpFeature(close, ma20, '-'),
            StdFeature(close, 20),
            '/'
        )
        self.add_feature(dev_20, 'alpha_015')
        
        # === Alpha 8: 振幅相关 ===
        # 日振幅
        amplitude = BinaryOpFeature(
            BinaryOpFeature(high, low, '-'),
            close,
            '/'
        )
        # 20日平均振幅
        self.add_feature(MeanFeature(amplitude, 20), 'alpha_016')
        
        # === Alpha 9: 量价相关 ===
        # 价格变化与成交量相关性（20日）
        close_change = DeltaFeature(close, 1)
        
        # 简化版：价格变化 * 成交量变化
        volume_change = DeltaFeature(volume, 1)
        self.add_feature(
            BinaryOpFeature(close_change, BinaryOpFeature(volume_change, FieldFeature('volume'), '/'), '*'),
            'alpha_017'
        )
        
        # === Alpha 10: 截面排名特征 ===
        # 当日收益率排名
        self.add_feature(RankFeature(ret_1), 'alpha_018')
        # 5日收益率排名
        self.add_feature(RankFeature(ReturnsFeature(close, 5)), 'alpha_019')
        # 成交量排名
        self.add_feature(RankFeature(volume), 'alpha_020')
        
        # === Alpha 11: 技术指标变体 ===
        # RSI 简化版（5日）
        gains = BinaryOpFeature(DeltaFeature(close, 1), FieldFeature('close'), '/')
        # 使用 rank 作为简化
        self.add_feature(RankFeature(MeanFeature(gains, 5)), 'alpha_021')
        
        # 价格加速度（二阶差分）
        self.add_feature(DeltaFeature(DeltaFeature(close, 5), 5), 'alpha_022')
        
        # === Alpha 12: 交叉特征 ===
        # 收益率 * 波动率
        self.add_feature(
            BinaryOpFeature(ReturnsFeature(close, 5), StdFeature(close, 20), '*'),
            'alpha_023'
        )
        
        # 价格位置 * 成交量变化
        self.add_feature(
            BinaryOpFeature(pos_20, ReturnsFeature(volume, 5), '*'),
            'alpha_024'
        )
        
        # === Alpha 13: 时序排名 ===
        # 5日收盘价在20日区间的排名位置
        close_rank_20 = BinaryOpFeature(
            BinaryOpFeature(close, MinFeature(RefFeature(close, 5), 20), '-'),
            BinaryOpFeature(
                MaxFeature(RefFeature(close, 5), 20),
                MinFeature(RefFeature(close, 5), 20),
                '-'
            ),
            '/'
        )
        self.add_feature(close_rank_20, 'alpha_025')
        
        # === Alpha 14: 趋势强度 ===
        # 短期趋势 vs 长期趋势
        trend_ratio = BinaryOpFeature(
            BinaryOpFeature(close, RefFeature(close, 5), '-'),
            BinaryOpFeature(close, RefFeature(close, 20), '-'),
            '/'
        )
        self.add_feature(trend_ratio, 'alpha_026')
        
        # === Alpha 15: 开盘收盘关系 ===
        # 开盘/收盘比率
        self.add_feature(BinaryOpFeature(open_p, close, '/'), 'alpha_027')
        # (收盘-开盘)/开盘
        self.add_feature(
            BinaryOpFeature(BinaryOpFeature(close, open_p, '-'), open_p, '/'),
            'alpha_028'
        )
        
        # === Alpha 16: 高低价关系 ===
        # 收盘价在当日高低区间的位置
        self.add_feature(
            BinaryOpFeature(
                BinaryOpFeature(close, low, '-'),
                BinaryOpFeature(high, low, '-'),
                '/'
            ),
            'alpha_029'
        )
        
        # === Alpha 17: 波动率调整收益率 ===
        # 夏普比率简化版（5日收益/5日波动）
        self.add_feature(
            BinaryOpFeature(
                ReturnsFeature(close, 5),
                StdFeature(close, 5),
                '/'
            ),
            'alpha_030'
        )
        
        print(f"✅ 已构建 {len(self.feature_exprs)} 个 Alpha 特征")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有特征
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含原始数据的 DataFrame
            
        Returns:
        --------
        pd.DataFrame
            添加了特征列的 DataFrame
        """
        df = df.copy()
        feature_dict = {}
        
        # 预计算常用基础特征
        print("🔧 计算基础特征...")
        
        # 计算所有特征表达式
        print(f"🔧 计算 {len(self.feature_exprs)} 个 Alpha 特征...")
        for expr, name in zip(self.feature_exprs, self.feature_names):
            try:
                df[name] = expr.evaluate(df, feature_dict)
            except Exception as e:
                print(f"   ⚠️ 特征 {name} 计算失败: {e}")
                df[name] = np.nan
        
        # 处理无穷值和缺失值
        alpha_cols = [c for c in df.columns if c.startswith('alpha_')]
        df[alpha_cols] = df[alpha_cols].replace([np.inf, -np.inf], np.nan)
        
        print(f"✅ 特征计算完成，共 {len(alpha_cols)} 个特征")
        return df
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        return self.feature_names.copy()


class QlibFeatureEngineerV2(QlibFeatureEngineer):
    """
    Qlib 特征工程器 V2 - 更丰富的特征集
    """
    
    def build_advanced_features(self) -> 'QlibFeatureEngineerV2':
        """
        构建高级 Alpha 特征
        """
        close = FieldFeature('close')
        open_p = FieldFeature('open')
        high = FieldFeature('high')
        low = FieldFeature('low')
        volume = FieldFeature('volume')
        
        # === 高级趋势特征 ===
        # 多时间框架趋势一致性
        for w in [5, 10, 20]:
            ma = MeanFeature(close, w)
            self.add_feature(BinaryOpFeature(close, ma, '>'), f'trend_above_ma{w}')
        
        # 价格突破
        high_20 = MaxFeature(RefFeature(high, 1), 20)
        low_20 = MinFeature(RefFeature(low, 1), 20)
        self.add_feature(BinaryOpFeature(close, high_20, '>'), 'break_high_20')
        self.add_feature(BinaryOpFeature(close, low_20, '<'), 'break_low_20')
        
        # === 成交量特征 ===
        # 放量/缩量
        vol_ma20 = MeanFeature(volume, 20)
        self.add_feature(BinaryOpFeature(volume, vol_ma20, '>'), 'volume_spike')
        
        # 价量背离（简化）
        price_change = ReturnsFeature(close, 5)
        vol_change = ReturnsFeature(volume, 5)
        self.add_feature(BinaryOpFeature(price_change, vol_change, '>'), 'price_vol_divergence')
        
        # === 波动率特征 ===
        # 布林带宽度
        bb_width = BinaryOpFeature(
            StdFeature(close, 20),
            MeanFeature(close, 20),
            '/'
        )
        self.add_feature(bb_width, 'bb_width')
        
        # ATR 简化版
        tr = BinaryOpFeature(high, low, '-')
        self.add_feature(MeanFeature(tr, 14), 'atr_14')
        
        # === 动量特征 ===
        # 不同时间框架动量差异
        mom5 = ReturnsFeature(close, 5)
        mom20 = ReturnsFeature(close, 20)
        self.add_feature(BinaryOpFeature(mom5, mom20, '-'), 'mom_diff_5_20')
        
        # 动量加速度
        self.add_feature(DeltaFeature(mom5, 5), 'mom_acceleration')
        
        # === 反转特征 ===
        # RSI 简化版
        gains = DeltaFeature(close, 1)
        avg_gain = MeanFeature(BinaryOpFeature(gains, FieldFeature('close'), '/'), 6)
        self.add_feature(avg_gain, 'rsi_proxy')
        
        # 威廉指标简化版
        williams = BinaryOpFeature(
            BinaryOpFeature(MaxFeature(high, 14), close, '-'),
            BinaryOpFeature(MaxFeature(high, 14), MinFeature(low, 14), '-'),
            '/'
        )
        self.add_feature(williams, 'williams_r')
        
        # === 截面特征 ===
        # 截面Z-Score
        self.add_feature(ScaleFeature(ReturnsFeature(close, 5)), 'cs_zscore_ret5')
        self.add_feature(ScaleFeature(volume), 'cs_zscore_volume')
        
        # 截面排名变化
        rank_ret5 = RankFeature(ReturnsFeature(close, 5))
        rank_ret5_prev = RefFeature(rank_ret5, 5)
        self.add_feature(BinaryOpFeature(rank_ret5, rank_ret5_prev, '-'), 'rank_change_5d')
        
        print(f"✅ 已构建 {len(self.feature_exprs)} 个高级特征")
        return self


# 便捷函数
def create_alpha_features(df: pd.DataFrame, advanced: bool = False) -> pd.DataFrame:
    """
    快速创建 Alpha 特征
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始数据
    advanced : bool
        是否使用高级特征
        
    Returns:
    --------
    pd.DataFrame
        添加特征后的数据
    """
    if advanced:
        engineer = QlibFeatureEngineerV2()
        engineer.build_alpha_features().build_advanced_features()
    else:
        engineer = QlibFeatureEngineer()
        engineer.build_alpha_features()
    
    return engineer.transform(df)
