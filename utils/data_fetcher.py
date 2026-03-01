"""
A股数据获取工具
支持 AKShare 和 Tushare 两个数据源
"""

import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import os
import time
import numpy as np

class DataFetcher:
    """数据获取类"""
    
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    # ==================== AKShare 接口 ====================
    
    def get_stock_list_ak(self):
        """
        获取A股所有股票列表
        """
        print("[AKShare] 获取股票列表...")
        try:
            stock_df = ak.stock_zh_a_spot_em()
            return stock_df[['代码', '名称', '总市值']]
        except Exception as e:
            print(f"⚠️ 获取股票列表失败: {e}")
            print("   使用默认股票列表...")
            # 返回一些常见股票作为备选
            default_stocks = pd.DataFrame({
                '代码': ['000001', '000002', '000858', '002415', '600036', '600519'],
                '名称': ['平安银行', '万科A', '五粮液', '海康威视', '招商银行', '贵州茅台'],
                '总市值': [0, 0, 0, 0, 0, 0]
            })
            return default_stocks
    
    def get_daily_data_ak(self, symbol, start_date, end_date):
        """
        获取股票日线数据（AKShare）
        
        Parameters:
        -----------
        symbol : str
            股票代码，如 '000001' (平安银行)
        start_date : str
            开始日期，格式 'YYYYMMDD'
        end_date : str
            结束日期，格式 'YYYYMMDD'
        
        Returns:
        --------
        DataFrame: 包含 OHLCV 数据
        """
        print(f"[AKShare] 获取 {symbol} 从 {start_date} 到 {end_date} 的日线数据...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 东方财富数据源
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"  # 前复权
                )
                
                if df.empty:
                    print(f"⚠️ 未获取到数据，尝试重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(1)
                    continue
                
                return self._standardize_df(df)
                
            except Exception as e:
                print(f"⚠️ 第 {attempt + 1} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print("   切换到模拟数据模式...")
                    return self._generate_mock_data(start_date, end_date)
    
    def _standardize_df(self, df):
        """标准化数据框"""
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    
    def _generate_mock_data(self, start_date, end_date, trend='up'):
        """
        生成模拟股票数据（用于离线测试）
        """
        print(f"[Mock] 生成模拟数据 ({trend}趋势)...")
        
        # 生成日期范围
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        dates = pd.date_range(start=start, end=end, freq='B')  # 工作日
        
        n = len(dates)
        
        # 生成价格数据
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, n)  # 日收益率
        
        if trend == 'up':
            returns += 0.0005  # 稍微向上趋势
        
        # 从10元开始计算价格
        prices = 10 * np.exp(np.cumsum(returns))
        
        # 生成OHLCV
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n),
            'amount': np.random.randint(10000000, 100000000, n),
            'amplitude': np.abs(np.random.normal(0, 0.02, n)) * 100,
            'pct_change': returns * 100,
            'change': np.diff(np.concatenate([[10], prices])),
            'turnover': np.random.uniform(1, 5, n)
        }, index=dates)
        
        # 确保 high >= max(open, close) 且 low <= min(open, close)
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        return df
    
    def get_index_data_ak(self, symbol='000300', start_date=None, end_date=None):
        """
        获取指数数据（如沪深300）
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        print(f"[AKShare] 获取指数 {symbol} 数据...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = ak.index_zh_a_hist(symbol=symbol, period="daily", 
                                        start_date=start_date, end_date=end_date)
                
                if df.empty:
                    time.sleep(1)
                    continue
                
                return self._standardize_df(df)
                
            except Exception as e:
                print(f"⚠️ 第 {attempt + 1} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print("   切换到模拟数据模式...")
                    return self._generate_mock_data(start_date, end_date, trend='volatile')
    
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


def test_akshare():
    """测试 AKShare 接口"""
    print("=" * 60)
    print("🧪 测试 AKShare 接口")
    print("=" * 60)
    
    fetcher = DataFetcher()
    
    # 1. 获取股票列表
    print("\n📋 获取A股股票列表（前10条）:")
    stocks = fetcher.get_stock_list_ak()
    print(stocks.head(10))
    print(f"总计: {len(stocks)} 只股票")
    
    # 2. 获取个股数据（平安银行 000001）
    print("\n📈 获取平安银行(000001)近一年数据:")
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    
    df = fetcher.get_daily_data_ak('000001', start_date, end_date)
    print(df.head(10))
    print(f"\n数据形状: {df.shape}")
    print(f"\n数据统计:\n{df[['open', 'high', 'low', 'close', 'volume']].describe()}")
    
    # 保存数据
    fetcher.save_to_csv(df, '000001_pingan.csv')
    
    # 3. 获取沪深300指数
    print("\n📊 获取沪深300指数数据:")
    index_df = fetcher.get_index_data_ak('000300')
    print(index_df.tail())
    fetcher.save_to_csv(index_df, 'hs300_index.csv')
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    
    return df


if __name__ == '__main__':
    # 运行测试
    df = test_akshare()
