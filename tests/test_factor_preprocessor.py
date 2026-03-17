"""
FactorPreprocessor 模块测试
测试去极值、标准化、中性化等因子预处理功能
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.factor_preprocessor import FactorPreprocessor


class TestWinsorizeMAD:
    """测试 MAD 去极值方法"""
    
    def test_winsorize_mad_clips_extreme_values(self, sample_factor_series):
        """测试 MAD 方法截断极端值"""
        factor = sample_factor_series.copy()
        original_max = factor.max()
        original_min = factor.min()
        
        result = FactorPreprocessor.winsorize_mad(factor, n=3)
        
        # 验证极端值被截断
        assert result.max() < original_max, "最大值应被截断"
        assert result.min() > original_min, "最小值应被截断"
    
    def test_winsorize_mad_preserves_median(self, sample_factor_series):
        """测试 MAD 方法保持中位数不变"""
        factor = sample_factor_series.copy()
        original_median = factor.median()
        
        result = FactorPreprocessor.winsorize_mad(factor, n=3)
        
        # 验证中位数基本保持不变（由于截断可能影响，允许微小变化）
        assert abs(result.median() - original_median) < 0.01, "中位数应基本保持不变"
    
    def test_winsorize_mad_different_n_values(self, sample_factor_series):
        """测试不同 n 值对截断效果的影响"""
        factor = sample_factor_series.copy()
        
        result_n1 = FactorPreprocessor.winsorize_mad(factor, n=1)
        result_n5 = FactorPreprocessor.winsorize_mad(factor, n=5)
        
        # n=1 应该截断更多，范围更小
        assert result_n1.max() <= result_n5.max(), "n=1 应该截断更多（最大值更小）"
        assert result_n1.min() >= result_n5.min(), "n=1 应该截断更多（最小值更大）"


class TestWinsorizeStd:
    """测试标准差去极值方法"""
    
    def test_winsorize_std_clips_by_std(self, sample_factor_series):
        """测试标准差方法按标准差倍数截断"""
        factor = sample_factor_series.copy()
        mean = factor.mean()
        std = factor.std()
        n = 2
        
        result = FactorPreprocessor.winsorize_std(factor, n=n)
        
        # 验证所有值在 mean ± n*std 范围内
        upper = mean + n * std
        lower = mean - n * std
        assert (result <= upper).all(), f"所有值应 <= {upper}"
        assert (result >= lower).all(), f"所有值应 >= {lower}"
    
    def test_winsorize_std_normal_distribution(self):
        """测试标准差方法对正态分布数据的效果"""
        np.random.seed(42)
        factor = pd.Series(np.random.normal(0, 1, 1000))
        
        result = FactorPreprocessor.winsorize_std(factor, n=3)
        
        # 对于正态分布，3倍标准差应包含约 99.7% 的数据
        clipped_count = (result != factor).sum()
        clip_rate = clipped_count / len(factor)
        assert clip_rate < 0.01, "3倍标准差应只截断极少量数据"


class TestWinsorizeQuantile:
    """测试分位数去极值方法"""
    
    def test_winsorize_quantile_clips_by_percentile(self, sample_factor_series):
        """测试分位数方法按百分位截断"""
        factor = sample_factor_series.copy()
        lower_q, upper_q = 0.05, 0.95
        
        result = FactorPreprocessor.winsorize_quantile(factor, lower_q=lower_q, upper_q=upper_q)
        
        # 验证截断后的范围
        expected_lower = factor.quantile(lower_q)
        expected_upper = factor.quantile(upper_q)
        
        assert (result >= expected_lower).all(), f"所有值应 >= {expected_lower} 分位数"
        assert (result <= expected_upper).all(), f"所有值应 <= {expected_upper} 分位数"
    
    def test_winsorize_quantile_extreme_percentiles(self, sample_factor_series):
        """测试极端分位数（1%-99%）"""
        factor = sample_factor_series.copy()
        
        result = FactorPreprocessor.winsorize_quantile(factor, lower_q=0.01, upper_q=0.99)
        
        # 验证 1% 和 99% 分位数的值
        lower_bound = factor.quantile(0.01)
        upper_bound = factor.quantile(0.99)
        
        assert (result >= lower_bound).all()
        assert (result <= upper_bound).all()


class TestWinsorizeDF:
    """测试 DataFrame 多列去极值"""
    
    def test_winsorize_df_multiple_columns(self, sample_factor_df):
        """测试对多列同时进行去极值"""
        df = sample_factor_df.copy()
        factor_cols = ['momentum_20', 'rsi_6', 'volatility_20']
        
        result = FactorPreprocessor.winsorize_df(df, factor_cols, method='mad', n=3)
        
        # 验证所有因子列都被处理
        for col in factor_cols:
            assert col in result.columns
            # 验证非因子列未被修改
            pd.testing.assert_series_equal(result['market_cap'], df['market_cap'])
    
    def test_winsorize_df_different_methods(self, sample_factor_df):
        """测试不同去极值方法"""
        df = sample_factor_df.copy()
        factor_cols = ['momentum_20']
        
        result_mad = FactorPreprocessor.winsorize_df(df.copy(), factor_cols, method='mad')
        result_std = FactorPreprocessor.winsorize_df(df.copy(), factor_cols, method='std')
        result_q = FactorPreprocessor.winsorize_df(df.copy(), factor_cols, method='quantile')
        
        # 验证三种方法产生不同结果
        assert not result_mad['momentum_20'].equals(result_std['momentum_20'])
        assert not result_std['momentum_20'].equals(result_q['momentum_20'])


class TestStandardizeZScore:
    """测试 Z-Score 标准化方法"""
    
    def test_standardize_zscore_mean_zero(self, sample_factor_series):
        """测试标准化后均值为 0"""
        factor = sample_factor_series.copy()
        
        result = FactorPreprocessor.standardize_zscore(factor)
        
        # 验证均值接近 0
        assert abs(result.mean()) < 0.001, f"标准化后均值应为 0，实际为 {result.mean()}"
    
    def test_standardize_zscore_std_one(self, sample_factor_series):
        """测试标准化后标准差为 1"""
        factor = sample_factor_series.copy()
        
        result = FactorPreprocessor.standardize_zscore(factor)
        
        # 验证标准差接近 1
        assert abs(result.std() - 1) < 0.001, f"标准化后标准差应为 1，实际为 {result.std()}"
    
    def test_standardize_zscore_constant_values(self):
        """测试常数序列的标准化"""
        factor = pd.Series([5.0] * 100)
        
        result = FactorPreprocessor.standardize_zscore(factor)
        
        # 常数序列标准化后应为 0
        assert (result == 0).all(), "常数序列标准化后应为全 0"


class TestStandardizeRank:
    """测试秩标准化方法"""
    
    def test_standardize_rank_mean_zero(self, sample_factor_series):
        """测试秩标准化后均值为 0"""
        factor = sample_factor_series.copy()
        
        result = FactorPreprocessor.standardize_rank(factor)
        
        # 验证均值接近 0
        assert abs(result.mean()) < 0.001, f"秩标准化后均值应为 0，实际为 {result.mean()}"
    
    def test_standardize_rank_std_one(self, sample_factor_series):
        """测试秩标准化后标准差为 1"""
        factor = sample_factor_series.copy()
        
        result = FactorPreprocessor.standardize_rank(factor)
        
        # 验证标准差接近 1
        assert abs(result.std() - 1) < 0.001, f"秩标准化后标准差应为 1，实际为 {result.std()}"
    
    def test_standardize_rank_robust_to_outliers(self):
        """测试秩标准化对异常值的稳健性"""
        # 创建包含极端异常值的数据
        factor = pd.Series([1, 2, 3, 4, 5, 1000])  # 1000 是极端异常值
        
        result = FactorPreprocessor.standardize_rank(factor)
        
        # 秩标准化后，异常值 1000 的秩为 6，不会极端影响结果
        # 验证结果范围合理
        assert result.max() < 3, "秩标准化对异常值更稳健"
        assert result.min() > -3, "秩标准化对异常值更稳健"


class TestStandardizeMinMax:
    """测试 Min-Max 标准化方法"""
    
    def test_standardize_minmax_range(self, sample_factor_series):
        """测试 Min-Max 标准化范围"""
        factor = sample_factor_series.copy()
        scale = 1
        
        result = FactorPreprocessor.standardize_minmax(factor, scale=scale)
        
        # 验证范围在 [-scale, scale] 内
        assert result.max() <= scale + 0.001, f"最大值应 <= {scale}"
        assert result.min() >= -scale - 0.001, f"最小值应 >= -scale"
    
    def test_standardize_minmax_different_scales(self, sample_factor_series):
        """测试不同 scale 参数"""
        factor = sample_factor_series.copy()
        
        result_scale1 = FactorPreprocessor.standardize_minmax(factor, scale=1)
        result_scale5 = FactorPreprocessor.standardize_minmax(factor, scale=5)
        
        # scale=5 的范围应该是 scale=1 的 5 倍
        range1 = result_scale1.max() - result_scale1.min()
        range5 = result_scale5.max() - result_scale5.min()
        assert abs(range5 / range1 - 5) < 0.01, "scale=5 的范围应该是 scale=1 的 5 倍"


class TestStandardizeDF:
    """测试 DataFrame 多列标准化"""
    
    def test_standardize_df_zscore(self, sample_factor_df):
        """测试 Z-Score 标准化多列"""
        df = sample_factor_df.copy()
        factor_cols = ['momentum_20', 'rsi_6']
        
        result = FactorPreprocessor.standardize_df(df, factor_cols, method='zscore')
        
        # 验证每列标准化后均值为 0，标准差为 1
        for col in factor_cols:
            assert abs(result[col].mean()) < 0.001, f"{col} 均值应为 0"
            assert abs(result[col].std() - 1) < 0.001, f"{col} 标准差应为 1"
    
    def test_standardize_df_rank(self, sample_factor_df):
        """测试秩标准化多列"""
        df = sample_factor_df.copy()
        factor_cols = ['momentum_20']
        
        result = FactorPreprocessor.standardize_df(df, factor_cols, method='rank')
        
        # 验证标准化后均值为 0，标准差为 1
        assert abs(result['momentum_20'].mean()) < 0.001
        assert abs(result['momentum_20'].std() - 1) < 0.001
    
    def test_standardize_df_minmax(self, sample_factor_df):
        """测试 Min-Max 标准化多列"""
        df = sample_factor_df.copy()
        factor_cols = ['momentum_20']
        
        result = FactorPreprocessor.standardize_df(df, factor_cols, method='minmax', scale=1)
        
        # 验证范围在 [-1, 1] 内
        assert result['momentum_20'].max() <= 1.001
        assert result['momentum_20'].min() >= -1.001


class TestNeutralizeMarketCap:
    """测试市值中性化方法"""
    
    def test_neutralize_market_cap_reduces_correlation(self, sample_factor_series, sample_market_cap):
        """测试市值中性化降低与市值的相关性"""
        # 创建与市值相关的因子
        factor = sample_factor_series.copy()
        market_cap = sample_market_cap.copy()
        
        # 添加市值相关成分
        factor_with_bias = factor + 0.001 * np.log(market_cap)
        
        corr_before = factor_with_bias.corr(np.log(market_cap))
        
        result = FactorPreprocessor.neutralize_market_cap(factor_with_bias, market_cap)
        
        corr_after = result.corr(np.log(market_cap))
        
        # 验证相关性降低
        assert abs(corr_after) < abs(corr_before), "中性化后相关性应降低"
    
    def test_neutralize_market_cap_log_transform(self, sample_factor_series, sample_market_cap):
        """测试对数变换参数"""
        factor = sample_factor_series.copy()
        market_cap = sample_market_cap.copy()
        
        # 测试 log_transform=True
        result_log = FactorPreprocessor.neutralize_market_cap(factor, market_cap, log_transform=True)
        
        # 测试 log_transform=False
        result_nolog = FactorPreprocessor.neutralize_market_cap(factor, market_cap, log_transform=False)
        
        # 两种方法应产生不同结果
        assert not result_log.equals(result_nolog)
    
    def test_neutralize_market_cap_handles_nan(self, sample_factor_series, sample_market_cap):
        """测试处理 NaN 值"""
        factor = sample_factor_series.copy()
        market_cap = sample_market_cap.copy()
        
        # 插入 NaN
        factor.iloc[0] = np.nan
        market_cap.iloc[1] = np.nan
        
        result = FactorPreprocessor.neutralize_market_cap(factor, market_cap)
        
        # 验证结果不为空且 NaN 被正确处理
        assert len(result) == len(factor)
        # 有效位置的值应为数值
        valid_idx = factor.notna() & market_cap.notna()
        assert result[valid_idx].notna().all()


class TestNeutralizeIndustry:
    """测试行业中性化方法"""
    
    def test_neutralize_industry_removes_industry_effect(self, sample_factor_series, sample_industry_dummy):
        """测试行业中性化去除行业效应"""
        factor = sample_factor_series.copy()
        industry_dummy = sample_industry_dummy.copy()
        
        # 创建行业相关因子
        industry_effect = industry_dummy['industry_A'].values * 2 + \
                         industry_dummy['industry_B'].values * (-1)
        factor_with_industry = factor + industry_effect
        
        result = FactorPreprocessor.neutralize_industry(factor_with_industry, industry_dummy)
        
        # 验证结果有效
        assert len(result) == len(factor)
        assert result.notna().sum() > 0
    
    def test_neutralize_industry_handles_nan(self, sample_factor_series, sample_industry_dummy):
        """测试处理 NaN 值"""
        factor = sample_factor_series.copy()
        industry_dummy = sample_industry_dummy.copy()
        
        # 插入 NaN
        factor.iloc[0] = np.nan
        
        result = FactorPreprocessor.neutralize_industry(factor, industry_dummy)
        
        # 验证结果长度正确
        assert len(result) == len(factor)


class TestNeutralize:
    """测试综合中性化方法"""
    
    def test_neutralize_both(self, sample_factor_series, sample_market_cap, sample_industry_dummy):
        """测试同时进行市值和行业中性化"""
        factor = sample_factor_series.copy()
        market_cap = sample_market_cap.copy()
        industry_dummy = sample_industry_dummy.copy()
        
        result = FactorPreprocessor.neutralize(factor, market_cap=market_cap, industry_dummy=industry_dummy)
        
        # 验证结果有效
        assert len(result) == len(factor)
        assert result.notna().sum() > 0
    
    def test_neutralize_only_market_cap(self, sample_factor_series, sample_market_cap):
        """测试仅市值中性化"""
        factor = sample_factor_series.copy()
        
        result = FactorPreprocessor.neutralize(factor, market_cap=sample_market_cap, industry_dummy=None)
        
        assert len(result) == len(factor)
    
    def test_neutralize_only_industry(self, sample_factor_series, sample_industry_dummy):
        """测试仅行业中性化"""
        factor = sample_factor_series.copy()
        
        result = FactorPreprocessor.neutralize(factor, market_cap=None, industry_dummy=sample_industry_dummy)
        
        assert len(result) == len(factor)


class TestPreprocessPipeline:
    """测试完整预处理流程"""
    
    def test_preprocess_pipeline_full(self, sample_factor_df):
        """测试完整的预处理流程"""
        df = sample_factor_df.copy()
        factor_cols = ['momentum_20', 'rsi_6', 'volatility_20']
        
        result = FactorPreprocessor.preprocess_pipeline(
            df,
            factor_cols,
            winsorize_method='mad',
            standardize_method='zscore',
            neutralize_cap=True,
            market_cap_col='market_cap'
        )
        
        # 验证所有因子列都被处理
        for col in factor_cols:
            assert col in result.columns
            # 验证标准化后均值为 0，标准差为 1
            assert abs(result[col].mean()) < 0.01, f"{col} 均值应接近 0"
            assert abs(result[col].std() - 1) < 0.01, f"{col} 标准差应接近 1"
    
    def test_preprocess_pipeline_no_neutralize(self, sample_factor_df):
        """测试不带中性化的预处理流程"""
        df = sample_factor_df.copy()
        factor_cols = ['momentum_20']
        
        result = FactorPreprocessor.preprocess_pipeline(
            df,
            factor_cols,
            winsorize_method='mad',
            standardize_method='zscore',
            neutralize_cap=False
        )
        
        # 验证标准化效果
        assert abs(result['momentum_20'].mean()) < 0.01
        assert abs(result['momentum_20'].std() - 1) < 0.01
    
    def test_preprocess_pipeline_different_methods(self, sample_factor_df):
        """测试不同方法组合的预处理流程"""
        df = sample_factor_df.copy()
        factor_cols = ['momentum_20']
        
        # 测试不同组合
        methods = [
            ('mad', 'zscore'),
            ('std', 'rank'),
            ('quantile', 'minmax')
        ]
        
        for win_method, std_method in methods:
            result = FactorPreprocessor.preprocess_pipeline(
                df.copy(),
                factor_cols,
                winsorize_method=win_method,
                standardize_method=std_method
            )
            
            assert 'momentum_20' in result.columns
            assert len(result) == len(df)
