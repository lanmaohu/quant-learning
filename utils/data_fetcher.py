"""
A股数据获取工具（升级版）
支持 AKShare、Tushare、本地缓存、模拟数据等多种数据源
自动故障转移，保证离线也能学习
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

# 导入新的统一数据源
try:
    from utils.data_sources import UnifiedDataFetcher
except ImportError:
    from data_sources import UnifiedDataFetcher


class DataFetcher:
    """
    数据获取类（兼容旧接口，使用新的统一数据源）
    """
    
    def __init__(self, data_dir='./data', tushare_token=None):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 使用新的统一数据获取器
        self._fetcher = UnifiedDataFetcher(
            tushare_token=tushare_token,
            prefer_cache=True
        )
    
    # ==================== 股票数据获取 ====================
    
    def get_stock_list_ak(self):
        """
        获取A股股票列表
        """
        print("[DataFetcher] 获取股票列表...")
        return self._fetcher.get_stock_list()
    
    def get_daily_data_ak(self, symbol, start_date, end_date):
        """
        获取股票日线数据（兼容旧接口）
        
        会自动尝试：AKShare -> Tushare -> 模拟数据
        """
        print(f"[DataFetcher] 获取 {symbol} 从 {start_date} 到 {end_date} 的日线数据...")
        
        try:
            # 使用统一数据获取器
            df = self._fetcher.get_daily_data(
                symbol, 
                start_date, 
                end_date,
                use_mock_params={
                    'base_price': 10,
                    'annual_return': 0.08,
                    'annual_volatility': 0.25,
                    'seed': hash(symbol) % 10000  # 不同股票不同随机种子
                }
            )
            return df
            
        except Exception as e:
            print(f"⚠️ 获取数据失败: {e}")
            print("   强制使用模拟数据...")
            
            # 强制使用模拟数据
            from utils.data_sources import MockDataSource
            mock = MockDataSource()
            return mock.get_daily_data(
                symbol, start_date, end_date,
                base_price=10,
                annual_return=0.08,
                annual_volatility=0.25,
                seed=hash(symbol) % 10000
            )
    
    def get_index_data_ak(self, symbol='000300', start_date=None, end_date=None):
        """
        获取指数数据（如沪深300）
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        print(f"[DataFetcher] 获取指数 {symbol} 数据...")
        
        # 指数使用模拟数据（因为没有真实数据源）
        try:
            from utils.data_sources import MockDataSource
        except ImportError:
            from data_sources import MockDataSource
        mock = MockDataSource()
        
        # 指数参数（波动较小）
        return mock.get_daily_data(
            symbol, start_date, end_date,
            base_price=4000 if symbol == '000300' else 1000,
            annual_return=0.05,
            annual_volatility=0.18,
            seed=hash(symbol) % 10000
        )
    
    def get_multiple_stocks(self, symbols, start_date=None, end_date=None):
        """
        批量获取多只股票数据
        
        Parameters:
        -----------
        symbols : list
            股票代码列表，如 ['000001', '000002']
        
        Returns:
        --------
        dict : {symbol: DataFrame}
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        print(f"[DataFetcher] 批量获取 {len(symbols)} 只股票数据...")
        
        results = {}
        for symbol in symbols:
            try:
                df = self.get_daily_data_ak(symbol, start_date, end_date)
                results[symbol] = df
                time.sleep(0.5)  # 避免请求过快
            except Exception as e:
                print(f"   ⚠️ 获取 {symbol} 失败: {e}")
                continue
        
        print(f"✅ 成功获取 {len(results)}/{len(symbols)} 只股票")
        return results
    
    # ==================== 数据存储 ====================
    
    def save_to_csv(self, df, filename):
        """保存数据到CSV"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)
        print(f"✅ 数据已保存: {filepath}")
        return filepath
    
    def load_from_csv(self, filename):
        """从CSV加载数据"""
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    
    def save_to_pickle(self, df, filename):
        """保存数据到pickle（更快，保留数据类型）"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_pickle(filepath)
        print(f"✅ 数据已保存: {filepath}")
        return filepath
    
    def load_from_pickle(self, filename):
        """从pickle加载数据"""
        filepath = os.path.join(self.data_dir, filename)
        df = pd.read_pickle(filepath)
        return df


def test_data_fetcher():
    """测试数据获取器（演示多种数据源）"""
    print("=" * 70)
    print("🧪 测试升级后的数据获取器")
    print("=" * 70)
    
    fetcher = DataFetcher()
    
    # 1. 获取股票列表
    print("\n📋 获取A股股票列表:")
    stocks = fetcher.get_stock_list_ak()
    print(stocks.head(10))
    
    # 2. 获取单只股票（自动故障转移到模拟数据）
    print("\n📈 获取平安银行(000001)数据:")
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    
    df = fetcher.get_daily_data_ak('000001', start_date, end_date)
    print(f"\n   数据形状: {df.shape}")
    print(f"\n   数据预览:")
    print(df[['open', 'high', 'low', 'close', 'volume', 'pct_change']].head())
    
    print(f"\n   数据统计:")
    print(df[['open', 'high', 'low', 'close']].describe())
    
    # 3. 获取指数
    print("\n📊 获取沪深300指数:")
    index_df = fetcher.get_index_data_ak('000300')
    print(index_df.tail())
    
    # 4. 保存数据
    print("\n💾 保存数据到本地...")
    fetcher.save_to_csv(df, '000001_demo.csv')
    fetcher.save_to_pickle(df, '000001_demo.pkl')
    
    # 5. 批量获取
    print("\n📦 批量获取多只股票:")
    symbols = ['000001', '000002', '600519']
    multi_data = fetcher.get_multiple_stocks(symbols, start_date, end_date)
    
    for symbol, data in multi_data.items():
        print(f"   {symbol}: {len(data)} 条记录，区间收益 {(data['close'].iloc[-1]/data['close'].iloc[0]-1)*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("✅ 数据获取器测试完成！")
    print("   特点：自动故障转移、模拟数据支持、离线可用")
    print("=" * 70)
    
    return df


if __name__ == '__main__':
    test_data_fetcher()
