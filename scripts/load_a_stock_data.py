#!/usr/bin/env python
"""
A股历史数据加载和分析工具
用法: python scripts/load_a_stock_data.py /path/to/a_stock_data.csv
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime


class AStockDataLoader:
    """
    A股历史数据加载器
    """

    def __init__(self, data_path=None):
        if data_path is None:
            data_path = sys.argv[1] if len(sys.argv) > 1 else input("请输入CSV文件路径: ").strip()
        self.data_path = data_path
        self.df = None
        
    def load_data(self, nrows=None, stock_code=None, start_date=None, end_date=None):
        """
        加载数据
        
        Parameters:
        -----------
        nrows : int
            只加载前N行（用于快速测试）
        stock_code : str or list
            指定股票代码，如 '000001' 或 ['000001', '000002']
        start_date : str
            开始日期 '2020-01-01'
        end_date : str
            结束日期 '2026-12-31'
        """
        print(f"📊 加载数据: {self.data_path}")
        
        # 使用chunks处理大文件
        if nrows:
            self.df = pd.read_csv(self.data_path, nrows=nrows)
        else:
            self.df = pd.read_csv(self.data_path)
        
        # 转换日期
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 过滤数据
        if stock_code:
            if isinstance(stock_code, str):
                stock_code = [stock_code]
            self.df = self.df[self.df['stock_code'].astype(str).isin(stock_code)]
        
        if start_date:
            self.df = self.df[self.df['date'] >= start_date]
        
        if end_date:
            self.df = self.df[self.df['date'] <= end_date]
        
        print(f"✅ 加载完成: {len(self.df):,} 条记录")
        return self.df
    
    def get_stock_list(self):
        """获取股票列表"""
        if self.df is None:
            raise ValueError("请先调用 load_data()")
        return self.df[['stock_code', 'name']].drop_duplicates().sort_values('stock_code')
    
    def get_stock_data(self, stock_code):
        """获取单只股票数据"""
        if self.df is None:
            raise ValueError("请先调用 load_data()")
        data = self.df[self.df['stock_code'].astype(str) == str(stock_code)].copy()
        data.set_index('date', inplace=True)
        data.sort_index(inplace=True)
        return data
    
    def calculate_returns(self, stock_code, period=5):
        """计算收益率"""
        data = self.get_stock_data(stock_code)
        data[f'return_{period}d'] = data['close'].pct_change(period) * 100
        return data
    
    def get_date_range(self):
        """获取日期范围"""
        if self.df is None:
            raise ValueError("请先调用 load_data()")
        return self.df['date'].min(), self.df['date'].max()


def demo_analysis():
    """数据分析演示"""
    print("=" * 70)
    print("📈 A股历史数据分析")
    print("=" * 70)
    
    loader = AStockDataLoader()
    
    # 1. 加载全部数据（或指定股票）
    print("\n📊 加载全部数据...")
    loader.load_data()
    
    # 2. 基本信息
    date_min, date_max = loader.get_date_range()
    stocks = loader.get_stock_list()
    
    print(f"\n📋 基本信息:")
    print(f"   日期范围: {date_min.date()} ~ {date_max.date()}")
    print(f"   股票数量: {len(stocks)}")
    print(f"   总记录数: {len(loader.df):,}")
    
    # 3. 查看股票列表
    print(f"\n📈 股票列表示例:")
    print(stocks.head(20).to_string(index=False))
    
    # 4. 分析单只股票
    print(f"\n📊 分析平安银行(000001):")
    pingan = loader.get_stock_data('1')  # 或 '000001'
    print(f"   数据条数: {len(pingan)}")
    print(f"   起始价格: {pingan['close'].iloc[0]:.2f}")
    print(f"   最新价格: {pingan['close'].iloc[-1]:.2f}")
    print(f"   区间收益: {(pingan['close'].iloc[-1]/pingan['close'].iloc[0]-1)*100:.2f}%")
    print(f"   波动率(年化): {pingan['price_change_rate'].std() * np.sqrt(252):.2f}%")
    
    # 5. 最新市场概况
    print(f"\n🌟 最新交易日 ({date_max.date()}) 市场概况:")
    latest = loader.df[loader.df['date'] == date_max]
    
    up = (latest['price_change_rate'] > 0).sum()
    down = (latest['price_change_rate'] < 0).sum()
    flat = (latest['price_change_rate'] == 0).sum()
    
    print(f"   上涨: {up} 只 ({up/len(latest)*100:.1f}%)")
    print(f"   下跌: {down} 只 ({down/len(latest)*100:.1f}%)")
    print(f"   平盘: {flat} 只 ({flat/len(latest)*100:.1f}%)")
    
    # 涨跌幅排行
    print(f"\n📈 涨幅前10:")
    top_gainers = latest.nlargest(10, 'price_change_rate')[['stock_code', 'name', 'close', 'price_change_rate']]
    print(top_gainers.to_string(index=False))
    
    print(f"\n📉 跌幅前10:")
    top_losers = latest.nsmallest(10, 'price_change_rate')[['stock_code', 'name', 'close', 'price_change_rate']]
    print(top_losers.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ 分析完成！")
    print("=" * 70)
    
    return loader


if __name__ == '__main__':
    loader = demo_analysis()
