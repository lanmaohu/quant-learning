"""
因子有效性检验模块
Day 10: IC分析、分层回测、因子衰减
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

try:
    from utils.logger import get_logger
except ImportError:
    from logger import get_logger

logger = get_logger(__name__)


class FactorAnalyzer:
    """
    因子有效性分析器
    用于评估因子的预测能力和稳定性
    """
    
    def __init__(self):
        self.results = {}
    
    # ========== IC分析 ==========
    
    @staticmethod
    def calculate_ic(factor_values, forward_returns, method='spearman'):
        """
        计算IC（信息系数）
        
        Parameters:
        -----------
        factor_values : Series
            因子值（已预处理）
        forward_returns : Series
            未来收益率（通常T+1到T+5）
        method : str
            'spearman' - 斯皮尔曼秩相关（更稳健，推荐）
            'pearson' - 皮尔逊线性相关
        
        Returns:
        --------
        ic : float
            IC值
        p_value : float
            显著性检验
        """
        # 去除NaN
        mask = factor_values.notna() & forward_returns.notna()
        f = factor_values[mask]
        r = forward_returns[mask]
        
        if len(f) < 10:
            return np.nan, np.nan
        
        if method == 'spearman':
            ic, p_value = stats.spearmanr(f, r)
        else:
            ic, p_value = stats.pearsonr(f, r)
        
        return ic, p_value
    
    def calculate_ic_series(self, df, factor_col, return_col='return_5d', date_col='date'):
        """
        计算IC时间序列（用于评估IC稳定性）
        
        Parameters:
        -----------
        df : DataFrame
            包含多期数据的DataFrame
        factor_col : str
            因子列名
        return_col : str
            未来收益率列名
        date_col : str
            日期列名（用于分组）
        """
        ic_series = []
        
        for date, group in df.groupby(date_col):
            ic, p_value = self.calculate_ic(
                group[factor_col], 
                group[return_col]
            )
            ic_series.append({
                'date': date,
                'ic': ic,
                'p_value': p_value
            })
        
        ic_df = pd.DataFrame(ic_series)
        return ic_df
    
    @staticmethod
    def ic_statistics(ic_series):
        """
        IC统计指标
        """
        ic_clean = ic_series.dropna()
        
        stats_dict = {
            'ic_mean': ic_clean.mean(),
            'ic_std': ic_clean.std(),
            'ic_ir': ic_clean.mean() / ic_clean.std() if ic_clean.std() != 0 else np.nan,
            'ic_ratio_positive': (ic_clean > 0).sum() / len(ic_clean),
            'ic_ratio_significant': (ic_clean.abs() > 0.02).sum() / len(ic_clean),
            'ic_skew': ic_clean.skew(),
            'ic_kurt': ic_clean.kurtosis()
        }
        
        return stats_dict
    
    # ========== 分层回测 ==========
    
    @staticmethod
    def quantile_backtest(df, factor_col, return_col='return_5d', n_quantiles=5, date_col='date'):
        """
        分层回测：将股票按因子值分为N组，观察每组收益
        
        Parameters:
        -----------
        n_quantiles : int
            分层层数，通常5或10
        
        Returns:
        --------
        quantile_returns : DataFrame
            每层每期的收益率
        cumulative_returns : DataFrame
            累计收益率
        """
        results = []
        
        for date, group in df.groupby(date_col):
            # 按因子值分层
            group['quantile'] = pd.qcut(
                group[factor_col], 
                n_quantiles, 
                labels=[f'Q{i+1}' for i in range(n_quantiles)],
                duplicates='drop'
            )
            
            # 计算每层平均收益
            quantile_return = group.groupby('quantile')[return_col].mean()
            quantile_return['date'] = date
            results.append(quantile_return)
        
        quantile_df = pd.DataFrame(results)
        quantile_df.set_index('date', inplace=True)
        
        # 计算累计收益
        cumulative_df = (1 + quantile_df).cumprod()
        
        return quantile_df, cumulative_df
    
    @staticmethod
    def calculate_spread(quantile_returns, top='Q5', bottom='Q1'):
        """
        计算多空组合收益（Top - Bottom）
        """
        if top in quantile_returns.columns and bottom in quantile_returns.columns:
            spread = quantile_returns[top] - quantile_returns[bottom]
            return spread
        return None
    
    # ========== 因子自相关性与衰减 ==========
    
    @staticmethod
    def calculate_factor_autocorrelation(df, factor_col, date_col='date', lag=1):
        """
        计算因子自相关系数（评估因子稳定性）
        """
        autocorrs = []
        
        dates = sorted(df[date_col].unique())
        for i in range(len(dates) - lag):
            current_date = dates[i]
            future_date = dates[i + lag]
            
            current_data = df[df[date_col] == current_date][['code', factor_col]]
            future_data = df[df[date_col] == future_date][['code', factor_col]]
            
            merged = pd.merge(current_data, future_data, on='code', suffixes=('_t', '_t+1'))
            
            if len(merged) > 10:
                corr = merged[f'{factor_col}_t'].corr(merged[f'{factor_col}_t+1'])
                autocorrs.append(corr)
        
        return np.mean(autocorrs) if autocorrs else np.nan
    
    @staticmethod
    def calculate_ic_decay(df, factor_col, return_cols=['return_1d', 'return_5d', 'return_10d', 'return_20d'], date_col='date'):
        """
        计算IC衰减（因子预测能力的持续时间）
        """
        ic_decays = {}
        
        for ret_col in return_cols:
            ic_series = []
            for date, group in df.groupby(date_col):
                ic, _ = FactorAnalyzer.calculate_ic(
                    group[factor_col],
                    group[ret_col]
                )
                ic_series.append(ic)
            
            ic_decays[ret_col] = np.nanmean(ic_series)
        
        return ic_decays
    
    # ========== 综合因子评估报告 ==========
    
    def generate_factor_report(self, df, factor_col, return_col='return_5d', date_col='date', n_quantiles=5):
        """
        生成因子分析报告
        """
        logger.info("=" * 60)
        logger.info(f"📊 因子分析报告: {factor_col}")
        logger.info("=" * 60)
        
        # 1. IC分析
        logger.info("\n📈 IC分析:")
        ic_series = self.calculate_ic_series(df, factor_col, return_col, date_col)
        ic_stats = self.ic_statistics(ic_series['ic'])
        
        logger.info(f"   IC均值: {ic_stats['ic_mean']:.4f}")
        logger.info(f"   IC标准差: {ic_stats['ic_std']:.4f}")
        logger.info(f"   IR比率: {ic_stats['ic_ir']:.4f}")
        logger.info(f"   IC正占比: {ic_stats['ic_ratio_positive']:.2%}")
        logger.info(f"   IC显著占比(|IC|>0.02): {ic_stats['ic_ratio_significant']:.2%}")
        
        # 2. 分层回测
        logger.info("\n📊 分层回测:")
        quantile_returns, cumulative_returns = self.quantile_backtest(
            df, factor_col, return_col, n_quantiles, date_col
        )
        
        # 每层平均收益
        mean_returns = quantile_returns.mean()
        logger.info(f"\n   各层平均{return_col}:")
        for q in mean_returns.index:
            logger.info(f"      {q}: {mean_returns[q]:.4f}")
        
        # 多空组合
        spread = self.calculate_spread(quantile_returns)
        if spread is not None:
            logger.info(f"\n   多空组合(Q5-Q1):")
            logger.info(f"      平均收益: {spread.mean():.4f}")
            logger.info(f"      胜率: {(spread > 0).mean():.2%}")
            logger.info(f"      夏普比率: {spread.mean() / spread.std() if spread.std() != 0 else np.nan:.4f}")
        
        # 3. 单调性检验
        logger.info("\n🔄 单调性检验:")
        monotonic = all(mean_returns.diff().dropna() > 0) or all(mean_returns.diff().dropna() < 0)
        logger.info(f"   收益单调递增/递减: {'✅ 是' if monotonic else '❌ 否'}")
        
        # 保存结果
        self.results[factor_col] = {
            'ic_stats': ic_stats,
            'quantile_returns': quantile_returns,
            'cumulative_returns': cumulative_returns,
            'spread': spread
        }
        
        return self.results[factor_col]


# ============== 测试代码 ==============

def test_factor_analyzer():
    """测试因子分析"""
    print("=" * 60)
    print("🧪 测试因子有效性分析模块")
    print("=" * 60)
    
    # 创建模拟多股票多期数据
    np.random.seed(42)
    
    n_dates = 30
    n_stocks = 100
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
    
    # 创建有效因子：动量因子（假设存在动量效应）
    data = []
    for date in dates:
        # 创建股票代码
        stocks = [f'STOCK_{i:03d}' for i in range(n_stocks)]
        
        # 因子值（随机但有一定自相关）
        factor_values = np.random.randn(n_stocks)
        
        # 未来收益率（与因子正相关，模拟真实效应）
        # 添加噪声使IC不会太高
        forward_returns = 0.1 * factor_values + 0.02 * np.random.randn(n_stocks)
        
        for i, stock in enumerate(stocks):
            data.append({
                'date': date,
                'code': stock,
                'momentum_factor': factor_values[i],
                'return_5d': forward_returns[i],
                'return_10d': 0.08 * factor_values[i] + 0.03 * np.random.randn(),
                'return_20d': 0.05 * factor_values[i] + 0.05 * np.random.randn()
            })
    
    df = pd.DataFrame(data)
    
    # 分析因子
    analyzer = FactorAnalyzer()
    
    print("\n" + "-" * 40)
    print("📊 测试动量因子:")
    result = analyzer.generate_factor_report(
        df, 
        factor_col='momentum_factor',
        return_col='return_5d',
        date_col='date',
        n_quantiles=5
    )
    
    # 测试IC衰减
    print("\n" + "-" * 40)
    print("📉 IC衰减分析:")
    ic_decay = analyzer.calculate_ic_decay(
        df, 
        'momentum_factor',
        ['return_5d', 'return_10d', 'return_20d'],
        date_col='date'
    )
    for period, ic in ic_decay.items():
        print(f"   {period}: IC = {ic:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 因子分析测试完成！")
    print("=" * 60)
    
    return analyzer


if __name__ == '__main__':
    analyzer = test_factor_analyzer()
