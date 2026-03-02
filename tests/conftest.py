"""
pytest 配置文件
提供测试 fixtures
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_price_df():
    """
    提供模拟的价格数据 DataFrame
    
    Returns:
        pd.DataFrame: 包含 OHLCV 数据的 DataFrame，索引为日期
    """
    dates = pd.date_range('2024-01-01', periods=30, freq='B')
    np.random.seed(42)
    
    # 生成基础价格
    base_price = 100
    changes = np.random.randn(30) * 0.02
    close_prices = base_price * (1 + changes).cumprod()
    
    # 生成 OHLC 数据
    data = {
        'open': close_prices * (1 + np.random.randn(30) * 0.005),
        'high': close_prices * (1 + np.abs(np.random.randn(30)) * 0.01 + 0.005),
        'low': close_prices * (1 - np.abs(np.random.randn(30)) * 0.01 - 0.005),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, 30)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # 确保 OHLC 逻辑一致性
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def sample_price_df_with_outliers():
    """
    提供包含异常值的价格数据 DataFrame
    
    Returns:
        pd.DataFrame: 包含异常值的价格数据
    """
    dates = pd.date_range('2024-01-01', periods=30, freq='B')
    np.random.seed(42)
    
    data = {
        'open': 10 + np.random.randn(30).cumsum() * 0.5,
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000000, 10000000, 30)
    }
    
    # 设置 OHLC 关系
    data['close'] = data['open'] + np.random.randn(30) * 0.3
    data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(30)) * 0.2
    data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(30)) * 0.2
    
    df = pd.DataFrame(data, index=dates)
    
    # 插入异常值
    df.loc[df.index[15], 'close'] = df.loc[df.index[15], 'close'] * 2.5  # 异常大涨
    df.loc[df.index[10], 'volume'] = 0  # 停牌
    
    return df


@pytest.fixture
def sample_factor_df():
    """
    提供模拟的因子数据 DataFrame
    
    Returns:
        pd.DataFrame: 包含多个因子列的 DataFrame
    """
    np.random.seed(42)
    n = 100
    
    data = {
        'momentum_20': np.random.normal(0.05, 0.15, n),
        'rsi_6': np.random.normal(50, 15, n),
        'volatility_20': np.random.exponential(0.3, n),
        'market_cap': np.random.lognormal(15, 1.5, n),
        'industry_code': np.random.choice(['A', 'B', 'C', 'D'], n)
    }
    
    df = pd.DataFrame(data)
    
    # 插入一些异常值
    df.loc[10, 'momentum_20'] = 5.0  # 极端正异常
    df.loc[20, 'momentum_20'] = -3.0  # 极端负异常
    df.loc[30, 'rsi_6'] = 200  # 超出理论范围
    
    return df


@pytest.fixture
def sample_factor_series():
    """
    提供模拟的因子 Series
    
    Returns:
        pd.Series: 单因子数据
    """
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    data[0] = 5.0  # 异常值
    data[1] = -4.0  # 异常值
    return pd.Series(data, name='test_factor')


@pytest.fixture
def sample_market_cap():
    """
    提供模拟的市值数据
    
    Returns:
        pd.Series: 市值数据
    """
    np.random.seed(42)
    return pd.Series(np.random.lognormal(15, 1.5, 100), name='market_cap')


@pytest.fixture
def sample_industry_dummy():
    """
    提供模拟的行业哑变量矩阵
    
    Returns:
        pd.DataFrame: 行业哑变量（one-hot编码）
    """
    np.random.seed(42)
    n = 100
    industries = np.random.choice(['A', 'B', 'C', 'D'], n)
    
    # 创建哑变量矩阵
    dummy = pd.get_dummies(industries, prefix='industry')
    return dummy
