"""
因子预处理模块
Day 9: 去极值(Winsorize)、标准化(Standardize)、中性化(Neutralize)
这是多因子模型中至关重要的一步
"""

import pandas as pd
import numpy as np
from scipy import stats


class FactorPreprocessor:
    """
    因子预处理器
    处理因子数据，使其更适合用于模型或策略
    """
    
    # ========== 去极值 (Winsorization) ==========
    
    @staticmethod
    def winsorize_mad(factor_series, n=3):
        """
        MAD去极值（中位数绝对偏差）- 更稳健
        
        Parameters:
        -----------
        n : int
            MAD倍数，默认3倍
        """
        median = factor_series.median()
        mad = np.median(np.abs(factor_series - median))
        
        # 调整后的z-score边界
        upper = median + n * 1.4826 * mad
        lower = median - n * 1.4826 * mad
        
        return factor_series.clip(lower, upper)
    
    @staticmethod
    def winsorize_std(factor_series, n=3):
        """
        标准差去极值
        """
        mean = factor_series.mean()
        std = factor_series.std()
        
        upper = mean + n * std
        lower = mean - n * std
        
        return factor_series.clip(lower, upper)
    
    @staticmethod
    def winsorize_quantile(factor_series, lower_q=0.01, upper_q=0.99):
        """
        分位数去极值
        """
        lower = factor_series.quantile(lower_q)
        upper = factor_series.quantile(upper_q)
        
        return factor_series.clip(lower, upper)
    
    @staticmethod
    def winsorize_df(df, factor_cols, method='mad', **kwargs):
        """
        对DataFrame多列进行去极值
        """
        df = df.copy()
        
        for col in factor_cols:
            if method == 'mad':
                df[col] = FactorPreprocessor.winsorize_mad(df[col], **kwargs)
            elif method == 'std':
                df[col] = FactorPreprocessor.winsorize_std(df[col], **kwargs)
            elif method == 'quantile':
                df[col] = FactorPreprocessor.winsorize_quantile(df[col], **kwargs)
        
        return df
    
    # ========== 标准化 (Standardization) ==========
    
    @staticmethod
    def standardize_zscore(factor_series):
        """
        Z-Score标准化：均值为0，标准差为1
        """
        mean = factor_series.mean()
        std = factor_series.std()
        
        if std == 0:
            return factor_series - mean
        
        return (factor_series - mean) / std
    
    @staticmethod
    def standardize_rank(factor_series):
        """
        秩标准化：转换为排名后标准化（更稳健，处理异常值好）
        """
        ranks = factor_series.rank()
        return FactorPreprocessor.standardize_zscore(ranks)
    
    @staticmethod
    def standardize_minmax(factor_series, scale=1):
        """
        Min-Max标准化到 [-scale, scale] 区间
        """
        min_val = factor_series.min()
        max_val = factor_series.max()
        
        if max_val == min_val:
            return factor_series * 0
        
        return 2 * scale * (factor_series - min_val) / (max_val - min_val) - scale
    
    @staticmethod
    def standardize_df(df, factor_cols, method='zscore', **kwargs):
        """
        对DataFrame多列进行标准化
        """
        df = df.copy()
        
        for col in factor_cols:
            if method == 'zscore':
                df[col] = FactorPreprocessor.standardize_zscore(df[col])
            elif method == 'rank':
                df[col] = FactorPreprocessor.standardize_rank(df[col])
            elif method == 'minmax':
                df[col] = FactorPreprocessor.standardize_minmax(df[col], **kwargs)
        
        return df
    
    # ========== 中性化 (Neutralization) ==========
    
    @staticmethod
    def neutralize_market_cap(factor_series, market_cap, log_transform=True):
        """
        市值中性化：去除因子与市值的相关性
        
        Parameters:
        -----------
        factor_series : Series
            待中性化的因子
        market_cap : Series
            市值数据
        log_transform : bool
            是否对市值取对数（通常推荐）
        """
        # 去除NaN
        valid_idx = factor_series.notna() & market_cap.notna()
        f = factor_series[valid_idx]
        cap = market_cap[valid_idx]
        
        if log_transform:
            cap = np.log(cap)
        
        # 线性回归去除市值影响
        cap_with_const = np.column_stack([np.ones(len(cap)), cap])
        
        try:
            beta = np.linalg.lstsq(cap_with_const, f, rcond=None)[0]
            residual = f - cap_with_const @ beta
            
            # 放回原始Series
            result = factor_series.copy()
            result[valid_idx] = residual
            return result
        except:
            return factor_series
    
    @staticmethod
    def neutralize_industry(factor_series, industry_dummy):
        """
        行业中性化：去除行业固定效应
        
        Parameters:
        -----------
        industry_dummy : DataFrame
            行业哑变量矩阵（one-hot编码）
        """
        valid_idx = factor_series.notna()
        f = factor_series[valid_idx]
        dummy = industry_dummy[valid_idx]
        
        try:
            beta = np.linalg.lstsq(dummy, f, rcond=None)[0]
            residual = f - dummy @ beta
            
            result = factor_series.copy()
            result[valid_idx] = residual
            return result
        except:
            return factor_series
    
    @staticmethod
    def neutralize(factor_series, market_cap=None, industry_dummy=None):
        """
        综合中性化
        """
        result = factor_series.copy()
        
        if market_cap is not None:
            result = FactorPreprocessor.neutralize_market_cap(result, market_cap)
        
        if industry_dummy is not None:
            result = FactorPreprocessor.neutralize_industry(result, industry_dummy)
        
        return result
    
    # ========== 完整预处理流程 ==========
    
    @staticmethod
    def preprocess_pipeline(df, factor_cols, 
                          winsorize_method='mad', 
                          standardize_method='zscore',
                          neutralize_cap=False,
                          market_cap_col=None,
                          industry_col=None):
        """
        因子预处理完整流程
        
        默认流程：去极值 -> 中性化（可选）-> 标准化
        """
        print(f"🔄 开始预处理 {len(factor_cols)} 个因子...")
        df = df.copy()
        
        # 1. 去极值
        print(f"   1️⃣ 去极值 ({winsorize_method})...")
        df = FactorPreprocessor.winsorize_df(df, factor_cols, method=winsorize_method)
        
        # 2. 中性化（可选）
        if neutralize_cap and market_cap_col:
            print("   2️⃣ 市值中性化...")
            for col in factor_cols:
                df[col] = FactorPreprocessor.neutralize_market_cap(
                    df[col], df[market_cap_col]
                )
        
        # 3. 标准化
        print(f"   3️⃣ 标准化 ({standardize_method})...")
        df = FactorPreprocessor.standardize_df(df, factor_cols, method=standardize_method)
        
        print("✅ 预处理完成！")
        return df


# ============== 测试代码 ==============

def test_preprocessor():
    """测试因子预处理"""
    print("=" * 60)
    print("🧪 测试因子预处理模块")
    print("=" * 60)
    
    # 创建含异常值的模拟因子数据
    np.random.seed(42)
    n = 500
    
    data = {
        'momentum_20': np.random.normal(0.05, 0.15, n),
        'rsi_6': np.random.normal(50, 15, n),
        'volatility_20': np.random.exponential(0.3, n),
        'market_cap': np.random.lognormal(15, 1.5, n)  # 市值
    }
    
    # 插入一些异常值
    data['momentum_20'][10] = 5.0  # 极端正异常
    data['momentum_20'][20] = -3.0  # 极端负异常
    data['rsi_6'][30] = 200  # 超出理论范围
    
    df = pd.DataFrame(data)
    factor_cols = ['momentum_20', 'rsi_6', 'volatility_20']
    
    print("\n📊 原始数据统计:")
    print(df[factor_cols].describe())
    print(f"\n   异常值检测 (>3倍std):")
    for col in factor_cols:
        outliers = np.abs(df[col] - df[col].mean()) > 3 * df[col].std()
        print(f"   {col}: {outliers.sum()} 个")
    
    # 测试各种去极值方法
    print("\n" + "-" * 40)
    print("🔧 测试去极值方法:")
    
    preprocessor = FactorPreprocessor()
    
    # MAD方法
    print("\n   MAD去极值后:")
    df_mad = preprocessor.winsorize_df(df.copy(), factor_cols, method='mad', n=3)
    print(df_mad[factor_cols].describe())
    
    # 分位数方法
    print("\n   分位数(1%-99%)去极值后:")
    df_q = preprocessor.winsorize_df(df.copy(), factor_cols, method='quantile', lower_q=0.01, upper_q=0.99)
    print(df_q[factor_cols].describe())
    
    # 测试标准化
    print("\n" + "-" * 40)
    print("📏 测试标准化:")
    
    print("\n   Z-Score标准化:")
    df_std = preprocessor.standardize_df(df_mad.copy(), factor_cols, method='zscore')
    print(df_std[factor_cols].describe().round(4))
    
    print("\n   Rank标准化:")
    df_rank = preprocessor.standardize_df(df_mad.copy(), factor_cols, method='rank')
    print(df_rank[factor_cols].describe().round(4))
    
    # 测试中性化
    print("\n" + "-" * 40)
    print("⚖️  测试市值中性化:")
    
    # 创建与市值相关的因子
    df_test = df_mad.copy()
    df_test['factor_with_cap_bias'] = df_test['momentum_20'] + 0.0001 * np.log(df_test['market_cap'])
    
    print(f"\n   中性化前与市值的相关系数: {df_test['factor_with_cap_bias'].corr(np.log(df_test['market_cap'])):.4f}")
    
    df_test['factor_neutralized'] = preprocessor.neutralize_market_cap(
        df_test['factor_with_cap_bias'], df_test['market_cap']
    )
    
    print(f"   中性化后与市值的相关系数: {df_test['factor_neutralized'].corr(np.log(df_test['market_cap'])):.4f}")
    
    # 完整流程
    print("\n" + "-" * 40)
    print("🔄 完整预处理流程:")
    
    df_processed = preprocessor.preprocess_pipeline(
        df.copy(),
        factor_cols,
        winsorize_method='mad',
        standardize_method='zscore',
        neutralize_cap=True,
        market_cap_col='market_cap'
    )
    
    print("\n📊 最终因子统计:")
    print(df_processed[factor_cols].describe().round(4))
    
    print("\n" + "=" * 60)
    print("✅ 因子预处理测试完成！")
    print("=" * 60)
    
    return df_processed


if __name__ == '__main__':
    test_preprocessor()
