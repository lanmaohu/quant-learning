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
        
        # === 收益率因子 ===
        # 5日/10日/20日/60日收益率（动量信号）
        self.add_feature(ReturnsFeature(close, 5), 'returns_5d')
        self.add_feature(ReturnsFeature(close, 10), 'returns_10d')
        self.add_feature(ReturnsFeature(close, 20), 'returns_20d')
        self.add_feature(ReturnsFeature(close, 60), 'returns_60d')

        # === 均线趋势因子 ===
        ma5 = MeanFeature(close, 5)
        ma20 = MeanFeature(close, 20)
        # 价格/5日均线比率：> 1 表示短期强势
        self.add_feature(BinaryOpFeature(close, ma5, '/'), 'close_ma5_ratio')
        # 价格/20日均线比率：> 1 表示中期强势
        self.add_feature(BinaryOpFeature(close, ma20, '/'), 'close_ma20_ratio')
        # 5日/20日均线比率：均线多头排列信号
        self.add_feature(BinaryOpFeature(ma5, ma20, '/'), 'ma5_ma20_ratio')

        # === 波动率因子 ===
        # 20日价格绝对波动（标准差）
        self.add_feature(StdFeature(close, 20), 'close_std_20d')
        # 20日收益率波动率（风险度量）
        ret_1 = ReturnsFeature(close, 1)
        self.add_feature(StdFeature(ret_1, 20), 'ret1_std_20d')

        # === 成交量因子 ===
        vol_ma5 = MeanFeature(volume, 5)
        vol_ma20 = MeanFeature(volume, 20)
        # 量比（近5日/近20日均量）：> 1 表示近期放量
        self.add_feature(BinaryOpFeature(vol_ma5, vol_ma20, '/'), 'vol_ma5_ma20_ratio')
        # 5日成交量变化率
        self.add_feature(ReturnsFeature(volume, 5), 'vol_returns_5d')

        # === 价格位置因子 ===
        high_20 = MaxFeature(high, 20)
        low_20 = MinFeature(low, 20)
        # 收盘价在20日高低区间的位置（0=底部，1=顶部）
        pos_20 = BinaryOpFeature(
            BinaryOpFeature(close, low_20, '-'),
            BinaryOpFeature(high_20, low_20, '-'),
            '/'
        )
        self.add_feature(pos_20, 'price_pos_20d')

        # === 价格动量因子 ===
        # 5日/10日价格变化量（趋势延续信号）
        self.add_feature(DeltaFeature(close, 5), 'close_delta_5d')
        self.add_feature(DeltaFeature(close, 10), 'close_delta_10d')

        # === 均值回归因子 ===
        # 价格偏离20日均线的 Z-score（>2 超买，<-2 超卖）
        dev_20 = BinaryOpFeature(
            BinaryOpFeature(close, ma20, '-'),
            StdFeature(close, 20),
            '/'
        )
        self.add_feature(dev_20, 'close_dev_ma20')

        # === 振幅因子 ===
        # 日振幅 = (high-low)/close（衡量当日波动范围）
        amplitude = BinaryOpFeature(BinaryOpFeature(high, low, '-'), close, '/')
        # 20日平均振幅
        self.add_feature(MeanFeature(amplitude, 20), 'amplitude_mean_20d')

        # === 量价关系因子 ===
        # 价格变化 × 成交量变化比（量价共振度量）
        close_change = DeltaFeature(close, 1)
        volume_change = DeltaFeature(volume, 1)
        self.add_feature(
            BinaryOpFeature(close_change, BinaryOpFeature(volume_change, FieldFeature('volume'), '/'), '*'),
            'pv_correlation'
        )

        # === 截面排名因子（控制行业/市值偏差）===
        # 当日1日收益率截面排名
        self.add_feature(RankFeature(ret_1), 'ret1_rank')
        # 5日收益率截面排名
        self.add_feature(RankFeature(ReturnsFeature(close, 5)), 'ret5_rank')
        # 成交量截面排名
        self.add_feature(RankFeature(volume), 'vol_rank')

        # === 技术指标变体 ===
        # RSI 代理（5日）：rank(avg_gain) 近似 RSI
        gains = BinaryOpFeature(DeltaFeature(close, 1), FieldFeature('close'), '/')
        self.add_feature(RankFeature(MeanFeature(gains, 5)), 'rsi_proxy_5d')
        # 价格加速度（二阶差分，捕捉趋势变化速率）
        self.add_feature(DeltaFeature(DeltaFeature(close, 5), 5), 'close_acceleration')

        # === 交叉因子 ===
        # 收益率 × 波动率（高收益+低波动 → 高分）
        self.add_feature(
            BinaryOpFeature(ReturnsFeature(close, 5), StdFeature(close, 20), '*'),
            'ret5_vol20_cross'
        )
        # 价格位置 × 成交量变化（突破高位+放量 → 高分）
        self.add_feature(
            BinaryOpFeature(pos_20, ReturnsFeature(volume, 5), '*'),
            'pos20_volret5_cross'
        )

        # === 时序排名因子 ===
        # 近5日收盘在过去20日价格序列中的位置
        close_rank_20 = BinaryOpFeature(
            BinaryOpFeature(close, MinFeature(RefFeature(close, 5), 20), '-'),
            BinaryOpFeature(
                MaxFeature(RefFeature(close, 5), 20),
                MinFeature(RefFeature(close, 5), 20),
                '-'
            ),
            '/'
        )
        self.add_feature(close_rank_20, 'close_rank_pos_20d')

        # === 趋势强度因子 ===
        # 短期趋势/长期趋势比率（> 1 表示短期加速）
        trend_ratio = BinaryOpFeature(
            BinaryOpFeature(close, RefFeature(close, 5), '-'),
            BinaryOpFeature(close, RefFeature(close, 20), '-'),
            '/'
        )
        self.add_feature(trend_ratio, 'trend_ratio_5_20')

        # === 开收盘关系因子 ===
        # 开盘/收盘比率（< 1 表示尾盘强势）
        self.add_feature(BinaryOpFeature(open_p, close, '/'), 'open_close_ratio')
        # 日内收益率 = (close-open)/open（尾盘追涨 or 压低）
        self.add_feature(
            BinaryOpFeature(BinaryOpFeature(close, open_p, '-'), open_p, '/'),
            'intraday_return'
        )

        # === 高低价关系因子 ===
        # 收盘在当日高低区间的位置（0=接近低价，1=接近高价）
        self.add_feature(
            BinaryOpFeature(
                BinaryOpFeature(close, low, '-'),
                BinaryOpFeature(high, low, '-'),
                '/'
            ),
            'close_hl_pos'
        )

        # === 波动率调整收益因子 ===
        # 5日夏普比率代理（收益率/波动率，越高越好）
        self.add_feature(
            BinaryOpFeature(
                ReturnsFeature(close, 5),
                StdFeature(close, 5),
                '/'
            ),
            'ret5_std5_ratio'
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
        
        # 处理无穷值和缺失值（使用显式特征名称列表，不再依赖前缀匹配）
        feature_cols = [n for n in self.feature_names if n in df.columns]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        print(f"✅ 特征计算完成，共 {len(feature_cols)} 个特征")
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
