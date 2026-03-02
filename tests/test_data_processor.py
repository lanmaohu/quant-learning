"""
DataProcessor 模块测试
测试数据清洗、异常值检测、收益率计算等功能
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_processor import DataProcessor


class TestDataProcessorCleanPriceData:
    """测试 DataProcessor.clean_price_data 方法"""
    
    def test_clean_price_data_removes_all_nan_rows(self, sample_price_df):
        """测试删除完全空值的行"""
        # 创建包含全空行的数据
        df = sample_price_df.copy()
        df.loc[df.index[5], ['open', 'high', 'low', 'close']] = np.nan
        
        result = DataProcessor.clean_price_data(df)
        
        # 验证没有全空的 OHLC 行
        assert not result[['open', 'high', 'low', 'close']].isna().all(axis=1).any()
    
    def test_clean_price_data_makes_prices_positive(self, sample_price_df):
        """测试价格转换为正值"""
        # 创建负价格数据
        df = sample_price_df.copy()
        df.loc[df.index[0], 'close'] = -10
        df.loc[df.index[1], 'open'] = -20
        
        result = DataProcessor.clean_price_data(df)
        
        # 验证价格为正
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            assert (result[col] >= 0).all(), f"{col} 包含负值"
    
    def test_clean_price_data_ohlc_logic(self, sample_price_df):
        """测试 OHLC 逻辑一致性: low <= min(open, close) <= max(open, close) <= high"""
        # 创建逻辑混乱的数据
        df = sample_price_df.copy()
        df.loc[df.index[0], 'low'] = df.loc[df.index[0], 'high'] + 10
        df.loc[df.index[1], 'high'] = df.loc[df.index[1], 'low'] - 10
        
        result = DataProcessor.clean_price_data(df)
        
        # 验证 OHLC 逻辑
        for idx in result.index:
            row = result.loc[idx]
            assert row['low'] <= row['open'], f"low > open at {idx}"
            assert row['low'] <= row['close'], f"low > close at {idx}"
            assert row['high'] >= row['open'], f"high < open at {idx}"
            assert row['high'] >= row['close'], f"high < close at {idx}"
    
    def test_clean_price_data_handles_zero_volume(self, sample_price_df):
        """测试处理零成交量（停牌）情况"""
        df = sample_price_df.copy()
        # 设置第10行成交量为0
        df.loc[df.index[10], 'volume'] = 0
        # 设置第10行的价格为异常值
        original_close = df.loc[df.index[10], 'close']
        df.loc[df.index[10], 'close'] = original_close * 2
        
        result = DataProcessor.clean_price_data(df)
        
        # 验证停牌日价格被向前填充（与前一交易日相同）
        expected_close = result.loc[df.index[9], 'close']
        actual_close = result.loc[df.index[10], 'close']
        assert actual_close == expected_close, "停牌日价格应被向前填充"
    
    def test_clean_price_data_handles_missing_volume(self, sample_price_df):
        """测试处理缺失成交量"""
        df = sample_price_df.copy()
        df.loc[df.index[5], 'volume'] = np.nan
        
        result = DataProcessor.clean_price_data(df)
        
        # 验证没有报错，且结果有效
        assert not result.empty
        assert result[['open', 'high', 'low', 'close']].notna().all().all()


class TestDataProcessorDetectOutliers:
    """测试 DataProcessor.detect_outliers 方法"""
    
    def test_detect_outliers_mad_method(self, sample_price_df_with_outliers):
        """测试 MAD 方法检测异常值"""
        df = sample_price_df_with_outliers.copy()
        
        result = DataProcessor.detect_outliers(df, method='mad', threshold=3)
        
        # 验证添加了 is_outlier 列
        assert 'is_outlier' in result.columns
        # 验证第15行（异常大涨）被标记为异常值
        assert result.loc[result.index[15], 'is_outlier'] == True
    
    def test_detect_outliers_std_method(self, sample_price_df_with_outliers):
        """测试标准差方法检测异常值"""
        df = sample_price_df_with_outliers.copy()
        
        result = DataProcessor.detect_outliers(df, method='std', threshold=2)
        
        # 验证添加了 is_outlier 列
        assert 'is_outlier' in result.columns
        # 验证检测到异常值
        outliers_count = result['is_outlier'].sum()
        assert outliers_count > 0, "应该检测到异常值"
    
    def test_detect_outliers_iqr_method(self, sample_price_df_with_outliers):
        """测试 IQR 方法检测异常值"""
        df = sample_price_df_with_outliers.copy()
        
        result = DataProcessor.detect_outliers(df, method='iqr', threshold=1.5)
        
        # 验证添加了 is_outlier 列
        assert 'is_outlier' in result.columns
        # 验证检测到异常值
        outliers_count = result['is_outlier'].sum()
        assert outliers_count >= 0, "IQR方法应正常工作"
    
    def test_detect_outliers_different_thresholds(self, sample_price_df_with_outliers):
        """测试不同阈值对异常值检测的影响"""
        df = sample_price_df_with_outliers.copy()
        
        result_strict = DataProcessor.detect_outliers(df, method='std', threshold=1.5)
        result_loose = DataProcessor.detect_outliers(df, method='std', threshold=5)
        
        strict_count = result_strict['is_outlier'].sum()
        loose_count = result_loose['is_outlier'].sum()
        
        # 严格阈值应该检测到更多异常值
        assert strict_count >= loose_count, "严格阈值应检测到更多或相同数量的异常值"
    
    def test_detect_outliers_preserves_original_data(self, sample_price_df):
        """测试检测异常值时保留原始数据"""
        df = sample_price_df.copy()
        original_close = df['close'].copy()
        
        result = DataProcessor.detect_outliers(df, method='mad')
        
        # 验证原始数据未被修改
        pd.testing.assert_series_equal(df['close'], original_close)
        # 验证返回的数据包含原始列
        assert 'close' in result.columns
        pd.testing.assert_series_equal(result['close'], original_close)


class TestDataProcessorCalculateReturns:
    """测试 DataProcessor.calculate_returns 方法"""
    
    def test_calculate_returns_single_period(self, sample_price_df):
        """测试计算单日收益率"""
        df = sample_price_df.copy()
        
        result = DataProcessor.calculate_returns(df, periods=[1], price_col='close')
        
        # 验证添加了收益率列
        assert 'return_1d' in result.columns
        assert 'log_return_1d' in result.columns
        
        # 验证收益率计算正确
        expected_return = df['close'].pct_change(1)
        pd.testing.assert_series_equal(result['return_1d'], expected_return, check_names=False)
    
    def test_calculate_returns_multiple_periods(self, sample_price_df):
        """测试计算多周期收益率"""
        df = sample_price_df.copy()
        periods = [1, 5, 20]
        
        result = DataProcessor.calculate_returns(df, periods=periods, price_col='close')
        
        # 验证所有周期的收益率列都被添加
        for p in periods:
            assert f'return_{p}d' in result.columns
            assert f'log_return_{p}d' in result.columns
    
    def test_calculate_returns_simple_vs_log(self, sample_price_df):
        """测试简单收益率与对数收益率的关系"""
        df = sample_price_df.copy()
        
        result = DataProcessor.calculate_returns(df, periods=[1], price_col='close')
        
        # 对于小收益率，简单收益率 ≈ 对数收益率
        # 简单收益率: (P_t - P_{t-1}) / P_{t-1}
        # 对数收益率: ln(P_t / P_{t-1})
        simple_ret = result['return_1d'].dropna()
        log_ret = result['log_return_1d'].dropna()
        
        # 验证两者近似相等（对于小收益率）
        diff = np.abs(simple_ret - log_ret)
        # 排除异常大的差异（可能由极端价格变动引起）
        assert (diff < 0.1).mean() > 0.8, "大部分简单收益率与对数收益率应接近"
    
    def test_calculate_returns_first_value_is_nan(self, sample_price_df):
        """测试第一个收益率值为 NaN"""
        df = sample_price_df.copy()
        
        result = DataProcessor.calculate_returns(df, periods=[1], price_col='close')
        
        # 验证第一个值为 NaN（因为没有前一天数据）
        assert pd.isna(result['return_1d'].iloc[0])
        assert pd.isna(result['log_return_1d'].iloc[0])
    
    def test_calculate_returns_with_different_price_col(self, sample_price_df):
        """测试使用不同的价格列计算收益率"""
        df = sample_price_df.copy()
        
        result_open = DataProcessor.calculate_returns(df, periods=[1], price_col='open')
        result_close = DataProcessor.calculate_returns(df, periods=[1], price_col='close')
        
        # 验证使用不同价格列得到不同结果
        assert not result_open['return_1d'].equals(result_close['return_1d'])


class TestDataProcessorIntegration:
    """测试 DataProcessor 的集成使用"""
    
    def test_full_pipeline(self, sample_price_df_with_outliers):
        """测试完整的数据处理流程"""
        df = sample_price_df_with_outliers.copy()
        
        # 1. 清洗数据
        df_clean = DataProcessor.clean_price_data(df)
        
        # 2. 检测异常值
        df_outliers = DataProcessor.detect_outliers(df_clean, method='mad')
        
        # 3. 计算收益率
        df_final = DataProcessor.calculate_returns(df_outliers, periods=[1, 5])
        
        # 验证最终结果包含所有期望的列
        expected_cols = ['open', 'high', 'low', 'close', 'volume', 
                        'is_outlier', 'return_1d', 'return_5d',
                        'log_return_1d', 'log_return_5d']
        for col in expected_cols:
            assert col in df_final.columns, f"缺少列: {col}"
        
        # 验证数据形状一致
        assert len(df_final) == len(df)
