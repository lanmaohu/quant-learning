"""
因子计算模块
Day 8: 技术指标因子 + 基础财务因子
"""

import pandas as pd
import numpy as np


class TechnicalFactorCalculator:
    """技术指标因子计算器"""
    
    @staticmethod
    def moving_average(df, windows=[5, 10, 20, 60]):
        """
        均线系统因子
        """
        for w in windows:
            df[f'ma_{w}'] = df['close'].rolling(window=w).mean()
            # 价格/均线比率
            df[f'price_to_ma_{w}'] = df['close'] / df[f'ma_{w}']
            # 均线斜率（趋势强度）
            df[f'ma_{w}_slope'] = df[f'ma_{w}'].diff(5) / df[f'ma_{w}'].shift(5)
        
        # 均线排列（多头排列：短期>中期>长期）
        if all(f'ma_{w}' in df.columns for w in [5, 10, 20]):
            df['ma_bull_arrange'] = ((df['ma_5'] > df['ma_10']) & 
                                     (df['ma_10'] > df['ma_20'])).astype(int)
        
        return df
    
    @staticmethod
    def rsi(df, windows=[6, 12, 24]):
        """
        RSI相对强弱指标
        """
        for w in windows:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
            rs = gain / loss
            df[f'rsi_{w}'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def macd(df, fast=12, slow=26, signal=9):
        """
        MACD指标
        """
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd_dif'] = ema_fast - ema_slow
        df['macd_dea'] = df['macd_dif'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = 2 * (df['macd_dif'] - df['macd_dea'])
        
        # MACD金叉/死叉信号
        df['macd_golden_cross'] = ((df['macd_dif'] > df['macd_dea']) & 
                                   (df['macd_dif'].shift(1) <= df['macd_dea'].shift(1))).astype(int)
        df['macd_dead_cross'] = ((df['macd_dif'] < df['macd_dea']) & 
                                 (df['macd_dif'].shift(1) >= df['macd_dea'].shift(1))).astype(int)
        
        return df
    
    @staticmethod
    def bollinger_bands(df, window=20, num_std=2):
        """
        布林带因子
        """
        ma = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        
        df['bb_upper'] = ma + num_std * std
        df['bb_lower'] = ma - num_std * std
        df['bb_middle'] = ma
        
        # 布林带宽度（波动率）
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma
        
        # 价格在布林带中的位置
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 突破信号
        df['bb_break_upper'] = (df['close'] > df['bb_upper']).astype(int)
        df['bb_break_lower'] = (df['close'] < df['bb_lower']).astype(int)
        
        return df
    
    @staticmethod
    def volatility_factors(df, windows=[20, 60]):
        """
        波动率因子
        """
        returns = df['close'].pct_change()
        
        for w in windows:
            # 历史波动率
            df[f'volatility_{w}'] = returns.rolling(window=w).std() * np.sqrt(252)
            
            # 振幅
            df[f'amplitude_{w}'] = ((df['high'] - df['low']) / df['close']).rolling(window=w).mean()
        
        # 高低点距离
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        df['price_position_20'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        return df
    
    @staticmethod
    def volume_factors(df):
        """
        成交量因子
        """
        # 量比（当日成交量/前5日均量）
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=5).mean()
        
        # 成交额移动平均
        df['amount_ma_5'] = df['amount'].rolling(window=5).mean()
        df['amount_ma_20'] = df['amount'].rolling(window=20).mean()
        
        # 量价相关性（20日）
        df['volume_price_corr'] = df['close'].rolling(window=20).corr(df['volume'])
        
        # 价涨量增信号
        price_up = df['close'].diff() > 0
        volume_up = df['volume'].diff() > 0
        df['price_vol_up'] = (price_up & volume_up).astype(int)
        
        return df
    
    @staticmethod
    def momentum_factors(df, windows=[5, 10, 20, 60]):
        """
        动量因子
        """
        for w in windows:
            # 价格动量（N日涨幅）
            df[f'momentum_{w}'] = df['close'] / df['close'].shift(w) - 1
            
            # 12个月动量（252个交易日）
            if w == 252:
                df['momentum_12m'] = df[f'momentum_{w}']
        
        # 短期/长期动量比
        if all(c in df.columns for c in ['momentum_20', 'momentum_60']):
            df['momentum_st_lt'] = df['momentum_20'] / (df['momentum_60'] + 1e-8)
        
        return df


class FundamentalFactorCalculator:
    """
    财务因子计算器（需要财务数据支持）
    简化版本，实际应用中需要接入财务数据库
    """
    
    @staticmethod
    def calculate_valuation_factors(price_df, financial_df):
        """
        估值因子（需要价格和财务数据合并）
        
        Parameters:
        -----------
        price_df : DataFrame
            包含市值、股价的数据
        financial_df : DataFrame
            包含财务指标的数据
        """
        df = price_df.copy()
        
        # 假设financial_df包含：eps, bvps, sales_per_share等
        if 'eps' in financial_df.columns:
            df['pe_ttm'] = df['close'] / financial_df['eps']  # 市盈率
        
        if 'bvps' in financial_df.columns:
            df['pb'] = df['close'] / financial_df['bvps']  # 市净率
        
        if 'sales_per_share' in financial_df.columns:
            df['ps'] = df['close'] / financial_df['sales_per_share']  # 市销率
        
        # 市值（如果有总股本数据）
        if 'total_shares' in financial_df.columns:
            df['market_cap'] = df['close'] * financial_df['total_shares']
        
        return df
    
    @staticmethod
    def calculate_quality_factors(financial_df):
        """
        质量因子（盈利能力、财务稳健性）
        """
        df = financial_df.copy()
        
        # ROE（净资产收益率）
        if all(c in df.columns for c in ['net_profit', 'equity']):
            df['roe'] = df['net_profit'] / df['equity']
        
        # ROA（总资产收益率）
        if all(c in df.columns for c in ['net_profit', 'total_assets']):
            df['roa'] = df['net_profit'] / df['total_assets']
        
        # 毛利率
        if all(c in df.columns for c in ['gross_profit', 'revenue']):
            df['gross_margin'] = df['gross_profit'] / df['revenue']
        
        # 营业利润率
        if all(c in df.columns for c in ['operating_profit', 'revenue']):
            df['operating_margin'] = df['operating_profit'] / df['revenue']
        
        # 财务杠杆
        if all(c in df.columns for c in ['total_assets', 'equity']):
            df['financial_leverage'] = df['total_assets'] / df['equity']
        
        return df


class FactorPipeline:
    """
    因子计算流水线：一键计算所有因子
    """
    
    def __init__(self):
        self.tech_calc = TechnicalFactorCalculator()
        self.fund_calc = FundamentalFactorCalculator()
    
    def calculate_all_factors(self, df, include_fundamental=False):
        """
        计算全部因子
        """
        print("🔄 开始计算因子...")
        
        # 技术指标因子
        print("   - 计算均线系统...")
        df = self.tech_calc.moving_average(df)
        
        print("   - 计算RSI...")
        df = self.tech_calc.rsi(df)
        
        print("   - 计算MACD...")
        df = self.tech_calc.macd(df)
        
        print("   - 计算布林带...")
        df = self.tech_calc.bollinger_bands(df)
        
        print("   - 计算波动率因子...")
        df = self.tech_calc.volatility_factors(df)
        
        print("   - 计算成交量因子...")
        df = self.tech_calc.volume_factors(df)
        
        print("   - 计算动量因子...")
        df = self.tech_calc.momentum_factors(df)
        
        print(f"✅ 因子计算完成，共 {len(df.columns)} 列")
        
        return df
    
    def get_factor_list(self, df, exclude_price_volume=True):
        """
        获取因子名称列表
        """
        price_volume_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        if exclude_price_volume:
            factors = [c for c in df.columns if c not in price_volume_cols]
        else:
            factors = list(df.columns)
        
        return factors


# ============== 测试代码 ==============

def test_factor_calculator():
    """测试因子计算"""
    print("=" * 60)
    print("🧪 测试因子计算模块")
    print("=" * 60)
    
    # 创建模拟股票数据
    from utils.data_fetcher import DataFetcher
    
    fetcher = DataFetcher()
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y%m%d')
    
    print("\n📥 获取测试数据...")
    df = fetcher.get_daily_data_ak('000001', start_date, end_date)
    print(f"   获取到 {len(df)} 条数据")
    
    # 计算因子
    pipeline = FactorPipeline()
    df_factors = pipeline.calculate_all_factors(df)
    
    # 显示因子列表
    print("\n📋 计算的因子列表:")
    factor_list = pipeline.get_factor_list(df_factors)
    for i, factor in enumerate(factor_list, 1):
        print(f"   {i:2d}. {factor}")
    
    # 显示因子统计
    print("\n📊 因子统计（最近20日）:")
    recent_df = df_factors[factor_list].tail(20)
    print(recent_df.describe().round(4))
    
    # 显示最新因子值
    print("\n📈 最新一日因子值:")
    print(df_factors[factor_list].iloc[-1].round(4))
    
    print("\n" + "=" * 60)
    print("✅ 因子计算测试完成！")
    print("=" * 60)
    
    return df_factors


if __name__ == '__main__':
    df = test_factor_calculator()
