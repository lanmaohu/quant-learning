"""
多数据源支持模块
提供 AKShare、Tushare、本地CSV、Yahoo Finance 等多种数据源
带自动故障转移和离线缓存功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import json

try:
    from utils.logger import get_logger
except ImportError:
    from logger import get_logger

logger = get_logger(__name__)

from .constants import TRADING_DAYS_PER_YEAR


class DataSourceBase:
    """数据源基类"""
    
    def __init__(self, cache_dir='./data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, symbol, start_date, end_date, suffix='pkl'):
        """获取缓存文件路径"""
        cache_key = f"{symbol}_{start_date}_{end_date}.{suffix}"
        return os.path.join(self.cache_dir, cache_key)
    
    def _save_cache(self, df, cache_path):
        """保存数据到缓存"""
        df.to_pickle(cache_path)
        logger.info(f"   💾 数据已缓存: {cache_path}")
    
    def _load_cache(self, cache_path, max_age_days=7):
        """从缓存加载数据"""
        if not os.path.exists(cache_path):
            return None
        
        # 检查缓存是否过期
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        if file_age.days > max_age_days:
            logger.info(f"   ⏰ 缓存已过期 ({file_age.days} 天)")
            return None
        
        try:
            df = pd.read_pickle(cache_path)
            logger.info(f"   ✅ 从缓存加载数据: {len(df)} 条记录")
            return df
        except Exception:
            return None
    
    def get_daily_data(self, symbol, start_date, end_date):
        """子类必须实现"""
        raise NotImplementedError


class AKShareSource(DataSourceBase):
    """AKShare数据源"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "AKShare"
        try:
            import akshare as ak
            self.ak = ak
            self.available = True
        except ImportError:
            logger.warning("⚠️ AKShare 未安装")
            self.available = False
    
    def get_daily_data(self, symbol, start_date, end_date):
        """获取日线数据"""
        if not self.available:
            raise Exception("AKShare 不可用")
        
        cache_path = self._get_cache_path(symbol, start_date, end_date)
        
        # 尝试从缓存加载
        cached = self._load_cache(cache_path)
        if cached is not None:
            return cached
        
        # 从网络获取
        logger.info(f"[{self.name}] 获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = self.ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                
                if df.empty:
                    raise Exception("返回数据为空")
                
                df = self._standardize(df)
                self._save_cache(df, cache_path)
                return df
                
            except Exception as e:
                logger.warning(f"   ⚠️ 第 {attempt + 1} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                else:
                    raise Exception(f"获取数据失败: {e}")
    
    def _standardize(self, df):
        """标准化数据格式"""
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


class TushareSource(DataSourceBase):
    """Tushare数据源（需要Token）"""
    
    def __init__(self, token=None, **kwargs):
        super().__init__(**kwargs)
        self.name = "Tushare"
        self.token = token
        
        try:
            import tushare as ts
            self.ts = ts
            if token:
                self.ts.set_token(token)
                self.pro = self.ts.pro_api()
                self.available = True
            else:
                logger.warning("⚠️ Tushare Token 未设置")
                self.available = False
        except ImportError:
            logger.warning("⚠️ Tushare 未安装")
            self.available = False
    
    def get_daily_data(self, symbol, start_date, end_date):
        """获取日线数据"""
        if not self.available:
            raise Exception("Tushare 不可用")
        
        cache_path = self._get_cache_path(symbol, start_date, end_date)
        cached = self._load_cache(cache_path)
        if cached is not None:
            return cached
        
        logger.info(f"[{self.name}] 获取 {symbol} 数据...")
        
        # Tushare 格式: 000001.SZ
        if symbol.startswith('6'):
            ts_code = f"{symbol}.SH"
        else:
            ts_code = f"{symbol}.SZ"
        
        df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        if df.empty:
            raise Exception("返回数据为空")
        
        df = self._standardize(df)
        self._save_cache(df, cache_path)
        return df
    
    def _standardize(self, df):
        """标准化数据格式"""
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount',
            'pct_chg': 'pct_change',
            'change': 'change'
        })
        return df


class MockDataSource(DataSourceBase):
    """
    模拟数据源
    当所有真实数据源都不可用时使用
    基于随机游走模型生成真实感的股票数据
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "MockData"
        self.available = True
    
    def get_daily_data(self, symbol, start_date, end_date, 
                      base_price=100,
                      annual_return=0.10,
                      annual_volatility=0.25,
                      seed=None):
        """
        生成模拟股票数据
        
        Parameters:
        -----------
        base_price : float
            起始价格
        annual_return : float
            年化收益率（如0.1表示10%）
        annual_volatility : float
            年化波动率（如0.25表示25%）
        seed : int
            随机种子，保证可重复
        """
        logger.info(f"[{self.name}] 生成 {symbol} 的模拟数据...")
        
        # 日期范围
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        dates = pd.date_range(start=start, end=end, freq='B')  # 工作日
        
        n = len(dates)
        if seed is not None:
            np.random.seed(seed)
        
        # 计算日收益率参数
        daily_return = annual_return / TRADING_DAYS_PER_YEAR
        daily_vol = annual_volatility / np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # 生成对数收益率（几何布朗运动）
        log_returns = np.random.normal(daily_return, daily_vol, n)
        
        # 计算价格序列
        log_prices = np.cumsum(log_returns)
        prices = base_price * np.exp(log_prices)
        
        # 生成OHLC
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        
        # 基于收盘价生成其他价格
        daily_range = daily_vol * prices
        df['high'] = df['close'] + np.abs(np.random.randn(n)) * daily_range * 0.5
        df['low'] = df['close'] - np.abs(np.random.randn(n)) * daily_range * 0.5
        df['open'] = df['close'].shift(1)
        df.loc[df.index[0], 'open'] = df['close'].iloc[0] * (1 + np.random.randn() * 0.01)
        
        # 确保OHLC逻辑
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        # 涨跌幅
        df['pct_change'] = df['close'].pct_change() * 100
        df.loc[df.index[0], 'pct_change'] = 0
        
        # 生成成交量（与波动率相关）
        base_volume = 1000000
        df['volume'] = base_volume * (1 + np.random.exponential(0.5, n)) * (1 + np.abs(df['pct_change']) * 10)
        df['volume'] = df['volume'].astype(int)
        
        # 成交额
        df['amount'] = df['volume'] * df['close']
        
        # 涨跌额
        df['change'] = df['close'].diff()
        
        # 振幅
        df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100
        
        # 换手率（模拟）
        total_shares = 1000000000
        df['turnover'] = df['volume'] / total_shares * 100
        
        logger.info(f"   ✅ 生成完成: {n} 条记录，起始价 {base_price:.2f}，结束价 {prices[-1]:.2f}")
        logger.info(f"      区间收益: {(prices[-1]/prices[0]-1)*100:.2f}%")
        
        return df


class UnifiedDataFetcher:
    """
    统一数据获取器
    自动尝试多个数据源，带故障转移
    """
    
    # 股票代码库（常用A股）
    STOCK_UNIVERSE = {
        '000001': '平安银行',
        '000002': '万科A',
        '000858': '五粮液',
        '002415': '海康威视',
        '002594': '比亚迪',
        '300750': '宁德时代',
        '600036': '招商银行',
        '600519': '贵州茅台',
        '601318': '中国平安',
        '601012': '隆基绿能',
    }
    
    def __init__(self, tushare_token=None, prefer_cache=True):
        self.sources = []
        self.prefer_cache = prefer_cache
        
        # 初始化各数据源
        self.ak_source = AKShareSource()
        self.ts_source = TushareSource(token=tushare_token)
        self.mock_source = MockDataSource()
        
        # 按优先级排序
        self.sources = [
            ('akshare', self.ak_source),
            ('tushare', self.ts_source),
            ('mock', self.mock_source),
        ]
    
    def get_daily_data(self, symbol, start_date=None, end_date=None, 
                      prefer_source=None,
                      use_mock_params=None):
        """
        获取日线数据（自动故障转移）
        
        Parameters:
        -----------
        symbol : str
            股票代码，如 '000001'
        start_date : str, optional
            开始日期 'YYYYMMDD'，默认为2年前
        end_date : str, optional
            结束日期 'YYYYMMDD'，默认为今天
        prefer_source : str, optional
            优先使用的数据源 'akshare'/'tushare'/'mock'
        use_mock_params : dict, optional
            使用模拟数据时的参数
        """
        # 默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y%m%d')
        
        # 如果指定了优先源，调整顺序
        sources = self.sources.copy()
        if prefer_source:
            sources = sorted(sources, 
                           key=lambda x: 0 if x[0] == prefer_source else 1)
        
        # 依次尝试各数据源
        errors = []
        for name, source in sources:
            if not source.available:
                continue
            
            try:
                logger.info(f"🔄 尝试使用 {name} 获取数据...")
                
                if name == 'mock' and use_mock_params:
                    df = source.get_daily_data(symbol, start_date, end_date, 
                                              **use_mock_params)
                else:
                    df = source.get_daily_data(symbol, start_date, end_date)
                
                logger.info(f"✅ {name} 获取成功！")
                return df
                
            except Exception as e:
                error_msg = f"{name}: {str(e)[:50]}"
                errors.append(error_msg)
                logger.error(f"   ❌ {error_msg}")
                continue
        
        # 所有源都失败
        raise Exception(f"所有数据源都失败: {'; '.join(errors)}")
    
    def get_stock_list(self):
        """获取股票列表"""
        return pd.DataFrame({
            'code': list(self.STOCK_UNIVERSE.keys()),
            'name': list(self.STOCK_UNIVERSE.values())
        })


# ============== 使用示例 ==============

def demo_unified_fetcher():
    """演示统一数据获取器"""
    print("=" * 70)
    print("🚀 统一数据获取器演示")
    print("=" * 70)
    
    fetcher = UnifiedDataFetcher()
    
    # 1. 显示可用股票
    print("\n📋 可用股票列表:")
    stocks = fetcher.get_stock_list()
    print(stocks.to_string(index=False))
    
    # 2. 获取数据（自动故障转移到模拟数据）
    print("\n📈 获取数据（自动故障转移）:")
    
    # 尝试获取真实数据，失败则使用模拟数据
    try:
        df = fetcher.get_daily_data(
            '000001', 
            '20240101', 
            '20241231',
            use_mock_params={
                'base_price': 10,
                'annual_return': 0.15,
                'annual_volatility': 0.30,
                'seed': 42
            }
        )
        
        print(f"\n📊 数据预览（最后5行）:")
        print(df[['open', 'high', 'low', 'close', 'volume']].tail())
        
        print(f"\n📈 数据统计:")
        print(df[['open', 'high', 'low', 'close']].describe())
        
    except Exception as e:
        print(f"❌ 获取失败: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 演示完成！")
    print("=" * 70)


if __name__ == '__main__':
    demo_unified_fetcher()
