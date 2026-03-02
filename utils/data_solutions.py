"""
数据抓取问题解决方案大全
提供多种数据源和抓取方式，确保能获取真实数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

try:
    from utils.logger import get_logger
except ImportError:
    from logger import get_logger

logger = get_logger(__name__)


class DataSolutionManager:
    """
    数据解决方案管理器
    按优先级尝试各种数据源，直到成功
    """
    
    SOLUTIONS = [
        "1. 本地缓存数据",
        "2. Yahoo Finance (美股/港股/A股港股通)", 
        "3. Tushare Pro (A股专业数据)",
        "4. Baostock (A股免费数据)",
        "5. 聚宽数据 (JoinQuant)",
        "6. 手动下载CSV导入",
    ]
    
    @staticmethod
    def print_solutions():
        """打印所有解决方案"""
        logger.info("=" * 70)
        logger.info("📚 数据抓取问题解决方案（按推荐优先级排序）")
        logger.info("=" * 70)
        for solution in DataSolutionManager.SOLUTIONS:
            logger.info(f"   {solution}")
        logger.info("=" * 70)


# ========== 方案 1: Yahoo Finance ==========

class YahooFinanceSource:
    """
    Yahoo Finance 数据源
    优势：无需注册，全球股票，包括A股港股通标的
    限制：A股只有港股通股票，数据有15分钟延迟
    """
    
    def __init__(self):
        self.name = "Yahoo Finance"
        try:
            import yfinance as yf
            self.yf = yf
            self.available = True
        except ImportError:
            logger.warning("⚠️ yfinance 未安装，运行: pip install yfinance")
            self.available = False
    
    @staticmethod
    def ascode_to_yahoo(symbol):
        """
        A股代码转 Yahoo Finance 格式
        000001.SZ -> 000001.SZ (深市)
        600519.SH -> 600519.SS (沪市)
        """
        symbol = symbol.replace('.SS', '.SH')
        if symbol.startswith('6'):
            return f"{symbol[:6]}.SS"  # 沪市
        else:
            return f"{symbol[:6]}.SZ"  # 深市
    
    def get_daily_data(self, symbol, start_date, end_date):
        """获取日线数据"""
        if not self.available:
            raise Exception("yfinance 未安装")
        
        logger.info(f"[{self.name}] 尝试获取 {symbol}...")
        
        # 转换代码格式
        yahoo_symbol = self.ascode_to_yahoo(symbol)
        
        # 转换日期格式
        start = pd.Timestamp(start_date).strftime('%Y-%m-%d')
        end = (pd.Timestamp(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 下载数据
        ticker = self.yf.Ticker(yahoo_symbol)
        df = ticker.history(start=start, end=end)
        
        if df.empty:
            raise Exception(f"未找到 {symbol} 的数据")
        
        # 标准化列名
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Dividends': 'dividends',
            'Stock Splits': 'splits'
        })
        
        df.index = df.index.tz_localize(None)  # 去除时区
        
        # 计算额外字段
        df['amount'] = df['volume'] * df['close']
        df['pct_change'] = df['close'].pct_change() * 100
        df['change'] = df['close'].diff()
        df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100
        
        logger.info(f"   ✅ 成功获取 {len(df)} 条数据")
        return df


# ========== 方案 2: Baostock ==========

class BaostockSource:
    """
    Baostock 数据源
    优势：免费，A股全量数据，包括财务数据
    限制：需要登录（免费），数据有延迟
    官网：www.baostock.com
    """
    
    def __init__(self):
        self.name = "Baostock"
        try:
            import baostock as bs
            self.bs = bs
            self.available = True
            self._login()
        except ImportError:
            logger.warning("⚠️ baostock 未安装，运行: pip install baostock")
            self.available = False
    
    def _login(self):
        """登录 Baostock"""
        result = self.bs.login()
        if result.error_code != '0':
            logger.warning(f"   ⚠️ Baostock 登录失败: {result.error_msg}")
            self.available = False
        else:
            logger.info(f"   ✅ Baostock 登录成功")
    
    def get_daily_data(self, symbol, start_date, end_date):
        """获取日线数据"""
        if not self.available:
            raise Exception("Baostock 不可用")
        
        logger.info(f"[{self.name}] 获取 {symbol}...")
        
        # 转换代码格式
        if symbol.startswith('6'):
            code = f"sh.{symbol}"
        else:
            code = f"sz.{symbol}"
        
        # 转换日期格式
        start = pd.Timestamp(start_date).strftime('%Y-%m-%d')
        end = pd.Timestamp(end_date).strftime('%Y-%m-%d')
        
        # 查询数据
        rs = self.bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3"  # 前复权
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            raise Exception("返回数据为空")
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 转换数值类型
        numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 标准化列名
        df = df.rename(columns={
            'pctChg': 'pct_change',
            'turn': 'turnover',
            'preclose': 'pre_close'
        })
        
        df['change'] = df['close'] - df['pre_close']
        df['amplitude'] = (df['high'] - df['low']) / df['pre_close'] * 100
        
        logger.info(f"   ✅ 成功获取 {len(df)} 条数据")
        return df


# ========== 方案 3: 离线CSV数据 ==========

class OfflineCSVSource:
    """
    离线CSV数据源
    从本地文件加载，或手动下载的数据
    """
    
    def __init__(self, data_dir='./data/offline'):
        self.name = "Offline CSV"
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.available = True
    
    def get_daily_data(self, symbol, start_date, end_date):
        """从CSV加载数据"""
        filepath = os.path.join(self.data_dir, f"{symbol}.csv")
        
        if not os.path.exists(filepath):
            raise Exception(f"本地文件不存在: {filepath}")
        
        logger.info(f"[{self.name}] 加载 {symbol}...")
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # 日期过滤
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        df = df[(df.index >= start) & (df.index <= end)]
        
        logger.info(f"   ✅ 成功加载 {len(df)} 条数据")
        return df
    
    def create_sample_csv(self, symbol='000001'):
        """创建示例CSV文件，用户可以基于此格式导入自己的数据"""
        sample_data = {
            'date': pd.date_range('2024-01-01', periods=10, freq='B'),
            'open': [10.0, 10.2, 10.1, 10.3, 10.5, 10.4, 10.6, 10.7, 10.5, 10.8],
            'high': [10.3, 10.4, 10.4, 10.6, 10.7, 10.7, 10.8, 10.9, 10.7, 11.0],
            'low': [9.9, 10.0, 10.0, 10.2, 10.3, 10.3, 10.5, 10.6, 10.4, 10.6],
            'close': [10.2, 10.1, 10.3, 10.5, 10.4, 10.6, 10.7, 10.5, 10.8, 10.9],
            'volume': [1000000, 1200000, 1100000, 1300000, 1500000, 1400000, 1600000, 1800000, 1700000, 2000000]
        }
        
        df = pd.DataFrame(sample_data)
        df.set_index('date', inplace=True)
        
        filepath = os.path.join(self.data_dir, f"{symbol}_sample.csv")
        df.to_csv(filepath)
        
        logger.info(f"✅ 示例CSV已创建: {filepath}")
        logger.info("   您可以复制此文件，修改文件名和数据，然后导入自己的数据")
        
        return filepath


# ========== 终极方案: 智能选择器 ==========

class SmartDataFetcher:
    """
    智能数据获取器
    自动尝试所有可用方案，直到成功
    """
    
    def __init__(self):
        self.sources = {}
        self._init_sources()
    
    def _init_sources(self):
        """初始化所有数据源"""
        # 优先级 1: Yahoo Finance
        yf = YahooFinanceSource()
        if yf.available:
            self.sources['yahoo'] = yf
        
        # 优先级 2: Baostock
        bs = BaostockSource()
        if bs.available:
            self.sources['baostock'] = bs
        
        # 优先级 3: 离线CSV
        offline = OfflineCSVSource()
        self.sources['offline'] = offline
    
    def get_daily_data(self, symbol, start_date=None, end_date=None, prefer_source=None):
        """
        获取数据（自动选择最佳方案）
        
        Parameters:
        -----------
        prefer_source : str
            'yahoo' - Yahoo Finance（推荐美股/港股通）
            'baostock' - Baostock（推荐A股）
            'offline' - 本地文件
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 如果指定了优先源，先尝试
        sources_to_try = list(self.sources.items())
        if prefer_source and prefer_source in self.sources:
            sources_to_try = [(prefer_source, self.sources[prefer_source])] + \
                           [(k, v) for k, v in sources_to_try if k != prefer_source]
        
        errors = []
        for name, source in sources_to_try:
            try:
                logger.info(f"🔄 尝试使用 {name}...")
                df = source.get_daily_data(symbol, start_date, end_date)
                logger.info(f"✅ {name} 成功！")
                return df
            except Exception as e:
                error_msg = f"{name}: {str(e)[:50]}"
                errors.append(error_msg)
                logger.error(f"   ❌ {error_msg}")
                continue
        
        # 所有方案都失败
        logger.error("\n" + "=" * 70)
        logger.error("❌ 所有数据源都失败")
        logger.error("=" * 70)
        logger.error("\n建议解决方案:")
        logger.error("1. 安装 yfinance: pip install yfinance")
        logger.error("2. 安装 baostock: pip install baostock")
        logger.error("3. 手动下载CSV数据放到 ./data/offline/ 目录")
        logger.error("   文件命名格式: {股票代码}.csv，如 000001.csv")
        logger.error("   必需列: date, open, high, low, close, volume")
        logger.error("=" * 70)
        
        raise Exception(f"所有数据源都失败: {'; '.join(errors)}")


def install_data_packages():
    """安装数据获取需要的包"""
    import subprocess
    import sys
    
    packages = ['yfinance', 'baostock', 'requests']
    
    logger.info("📦 安装数据获取依赖包...")
    for pkg in packages:
        logger.info(f"   安装 {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    
    logger.info("✅ 安装完成！")


# ============== 测试 ==============

def test_all_solutions():
    """测试所有数据解决方案"""
    print("=" * 70)
    print("🚀 测试数据解决方案")
    print("=" * 70)
    
    # 显示所有方案
    DataSolutionManager.print_solutions()
    
    # 初始化智能获取器
    fetcher = SmartDataFetcher()
    
    print(f"\n📊 可用数据源: {list(fetcher.sources.keys())}")
    
    # 测试获取数据
    print("\n📈 尝试获取 000001 数据:")
    try:
        df = fetcher.get_daily_data('000001', '20240101', '20241231', prefer_source='yahoo')
        print(f"\n   数据形状: {df.shape}")
        print(f"   列: {list(df.columns)}")
        print(f"\n   数据预览:")
        print(df[['open', 'high', 'low', 'close', 'volume']].head())
    except Exception as e:
        print(f"   ⚠️ 获取失败: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    # 首次运行建议安装依赖
    # install_data_packages()
    test_all_solutions()
