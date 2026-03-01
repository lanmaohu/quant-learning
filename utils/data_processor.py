"""
数据预处理模块
Day 6-7: 数据清洗、复权处理、异常值处理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataProcessor:
    """数据处理器：清洗、预处理、格式标准化"""
    
    @staticmethod
    def clean_price_data(df):
        """
        清洗价格数据
        - 处理停牌日（向前填充）
        - 处理价格异常值
        - 确保OHLC逻辑一致性
        """
        df = df.copy()
        
        # 1. 删除完全空值的行
        df.dropna(subset=['open', 'high', 'low', 'close'], how='all', inplace=True)
        
        # 2. 价格必须为正
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = df[col].abs()
        
        # 3. 确保OHLC逻辑：low <= min(open, close) <= max(open, close) <= high
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        
        # 4. 处理零成交量（可能停牌）
        if 'volume' in df.columns:
            # 成交量为0或NaN视为停牌，价格向前填充
            mask = (df['volume'] == 0) | df['volume'].isna()
            df.loc[mask, price_cols] = np.nan
            df[price_cols] = df[price_cols].fillna(method='ffill')
        
        # 5. 删除连续停牌过多的股票（可选）
        
        return df
    
    @staticmethod
    def adjust_prices(df, adj_factor=None, method='forward'):
        """
        价格复权处理
        
        Parameters:
        -----------
        df : DataFrame
            包含价格数据的DataFrame
        adj_factor : Series, optional
            复权因子，如果为None则假设数据已复权
        method : str
            'forward' - 前复权（常用）
            'backward' - 后复权
        """
        if adj_factor is None:
            return df
        
        df = df.copy()
        price_cols = ['open', 'high', 'low', 'close']
        
        # 复权价格 = 原始价格 * 复权因子
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col] * adj_factor
        
        return df
    
    @staticmethod
    def detect_outliers(df, method='mad', threshold=3):
        """
        检测价格异常值
        
        Parameters:
        -----------
        method : str
            'mad' - 中位数绝对偏差（稳健）
            'std' - 标准差
            'iqr' - 四分位距
        """
        df = df.copy()
        
        if method == 'mad':
            # MAD: Median Absolute Deviation
            median = df['close'].median()
            mad = np.median(np.abs(df['close'] - median))
            modified_z = 0.6745 * (df['close'] - median) / mad
            outliers = np.abs(modified_z) > threshold
            
        elif method == 'std':
            mean = df['close'].mean()
            std = df['close'].std()
            outliers = np.abs(df['close'] - mean) > threshold * std
            
        elif method == 'iqr':
            Q1 = df['close'].quantile(0.25)
            Q3 = df['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = (df['close'] < lower) | (df['close'] > upper)
        
        df['is_outlier'] = outliers
        return df
    
    @staticmethod
    def resample_data(df, freq='W', agg_func=None):
        """
        数据重采样（日线转周线/月线）
        
        Parameters:
        -----------
        freq : str
            'W' - 周线
            'M' - 月线
            'Q' - 季线
        """
        if agg_func is None:
            agg_func = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            }
        
        # 只取存在的列
        agg_func = {k: v for k, v in agg_func.items() if k in df.columns}
        
        resampled = df.resample(freq).agg(agg_func)
        resampled.dropna(inplace=True)
        
        return resampled
    
    @staticmethod
    def fill_missing_dates(df, method='ffill'):
        """
        填充缺失交易日（补全日期索引）
        """
        # 创建完整的交易日历（简化版，实际应该使用交易日历）
        all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        
        df = df.reindex(all_dates)
        
        if method == 'ffill':
            df.fillna(method='ffill', inplace=True)
        elif method == 'interpolate':
            df.interpolate(method='linear', inplace=True)
        
        return df
    
    @staticmethod
    def calculate_returns(df, periods=[1, 5, 20], price_col='close'):
        """
        计算收益率
        
        Parameters:
        -----------
        periods : list
            计算收益率的周期，1=日收益，5=周收益，20=月收益
        """
        df = df.copy()
        
        for p in periods:
            df[f'return_{p}d'] = df[price_col].pct_change(p)
            # 对数收益率（便于计算）
            df[f'log_return_{p}d'] = np.log(df[price_col] / df[price_col].shift(p))
        
        return df


class DataStore:
    """
    数据存储管理（简单版，生产环境可用SQLite/PostgreSQL）
    """
    
    def __init__(self, base_path='./data'):
        self.base_path = base_path
        import os
        os.makedirs(base_path, exist_ok=True)
    
    def save_stock_data(self, df, symbol, data_type='daily'):
        """保存股票数据"""
        filepath = f"{self.base_path}/{symbol}_{data_type}.csv"
        df.to_csv(filepath)
        return filepath
    
    def load_stock_data(self, symbol, data_type='daily'):
        """加载股票数据"""
        filepath = f"{self.base_path}/{symbol}_{data_type}.csv"
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    
    def save_factor_data(self, df, factor_name):
        """保存因子数据"""
        filepath = f"{self.base_path}/factor_{factor_name}.csv"
        df.to_csv(filepath)
        return filepath


# ============== 测试代码 ==============

def test_data_processor():
    """测试数据预处理功能"""
    print("=" * 60)
    print("🧪 测试数据预处理模块")
    print("=" * 60)
    
    # 创建模拟数据（含异常值和缺失值）
    dates = pd.date_range('2024-01-01', periods=30, freq='B')
    np.random.seed(42)
    
    data = {
        'open': 10 + np.random.randn(30).cumsum() * 0.5,
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000000, 10000000, 30)
    }
    
    # 设置OHLC关系
    for i in range(30):
        base = data['open'][i]
        data['close'] = np.append(data['close'][:i], base + np.random.randn() * 0.3) if i > 0 else [base]
        
    data['close'] = data['open'] + np.random.randn(30) * 0.3
    data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(30)) * 0.2
    data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(30)) * 0.2
    
    # 插入一些异常值
    data['close'][15] = data['close'][15] * 2  # 异常大涨
    data['volume'][10] = 0  # 停牌
    
    df = pd.DataFrame(data, index=dates)
    
    print("\n📊 原始数据（前5行）:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    
    # 1. 清洗数据
    print("\n" + "-" * 40)
    print("🧹 数据清洗...")
    processor = DataProcessor()
    df_clean = processor.clean_price_data(df)
    print("✅ 清洗完成")
    print(df_clean.head())
    
    # 2. 异常值检测
    print("\n" + "-" * 40)
    print("🔍 异常值检测 (MAD方法)...")
    df_outlier = processor.detect_outliers(df_clean, method='mad')
    outliers = df_outlier[df_outlier['is_outlier']]
    print(f"发现 {len(outliers)} 个异常值:")
    print(outliers[['close', 'is_outlier']])
    
    # 3. 计算收益率
    print("\n" + "-" * 40)
    print("📈 计算收益率...")
    df_returns = processor.calculate_returns(df_clean, periods=[1, 5])
    print(df_returns[['close', 'return_1d', 'return_5d']].head(10))
    
    # 4. 数据重采样
    print("\n" + "-" * 40)
    print("📅 周线数据...")
    df_weekly = processor.resample_data(df_clean, freq='W')
    print(f"日线 {len(df_clean)} 条 -> 周线 {len(df_weekly)} 条")
    print(df_weekly.head())
    
    print("\n" + "=" * 60)
    print("✅ 数据预处理测试完成！")
    print("=" * 60)
    
    return df_clean


if __name__ == '__main__':
    test_data_processor()
