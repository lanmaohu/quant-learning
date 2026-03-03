#!/usr/bin/env python
"""
A股历史数据与量化框架集成
将外部CSV数据集成到我们的因子研究流程中
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_processor import DataProcessor
from utils.factor_calculator import FactorPipeline
from utils.factor_preprocessor import FactorPreprocessor
from utils.factor_analyzer import FactorAnalyzer


class AStockDataAdapter:
    """
    A股数据适配器
    将外部CSV数据转换为我们的量化框架格式
    """
    
    def __init__(self, data_path='/Users/harry/Documents/a_stock_history_price_20260223.csv'):
        self.data_path = data_path
        self.raw_df = None
        self.processed_df = None
        
    def load_and_transform(self, stock_code=None, nrows=None):
        """
        加载并转换数据格式
        
        Parameters:
        -----------
        stock_code : str
            指定股票代码，如 '000001'
        nrows : int
            限制加载行数
        """
        print(f"📊 加载数据: {self.data_path}")
        
        # 读取CSV
        df = pd.read_csv(self.data_path, nrows=nrows)
        
        # 转换日期
        df['date'] = pd.to_datetime(df['date'])
        
        # 转换股票代码格式
        df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        
        # 筛选指定股票
        if stock_code:
            df = df[df['stock_code'] == stock_code]
        
        # 标准化列名（匹配我们的框架）
        df = df.rename(columns={
            'stock_code': 'code',
            'price_change': 'change',
            'price_change_rate': 'pct_change',
            'turnover_rate': 'turnover',
            'market_cap': 'market_cap'
        })
        
        # 确保有 amount 列（成交额）
        if 'amount' not in df.columns:
            df['amount'] = df['volume'] * df['close']
        
        # 设置索引
        df = df.sort_values(['code', 'date'])
        
        self.raw_df = df
        print(f"✅ 加载完成: {len(df):,} 条记录")
        print(f"   股票数: {df['code'].nunique()}")
        print(f"   日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
        
        return df
    
    def get_single_stock(self, code):
        """获取单只股票的标准格式数据"""
        if self.raw_df is None:
            raise ValueError("请先调用 load_and_transform()")
        
        df = self.raw_df[self.raw_df['code'] == code].copy()
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 确保有必需的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必需列: {col}")
        
        return df
    
    def run_factor_pipeline(self, code):
        """
        对单只股票运行完整因子流程
        """
        print(f"\n📈 分析股票: {code}")
        print("=" * 60)
        
        # 1. 获取数据
        df = self.get_single_stock(code)
        print(f"1️⃣ 数据加载: {len(df)} 条")
        
        # 2. 数据预处理
        processor = DataProcessor()
        df = processor.clean_price_data(df)
        df = processor.calculate_returns(df, periods=[1, 5, 10, 20])
        print(f"2️⃣ 数据预处理完成")
        
        # 3. 计算因子（包含KDJ）
        factor_pipeline = FactorPipeline()
        df = factor_pipeline.calculate_all_factors(df)
        
        # 额外计算KDJ
        from utils.factor_calculator import TechnicalFactorCalculator
        tech_calc = TechnicalFactorCalculator()
        df = tech_calc.kdj(df)
        print(f"3️⃣ 因子计算完成，包含KDJ指标 (K/D/J)")
        
        factor_cols = factor_pipeline.get_factor_list(df)
        factor_cols.extend(['kdj_k', 'kdj_d', 'kdj_j'])  # 添加KDJ列
        
        # 过滤出数值型因子列
        factor_cols = [c for c in factor_cols if pd.api.types.is_numeric_dtype(df[c])]
        print(f"   共 {len(factor_cols)} 个因子")
        
        # 4. 因子预处理
        preprocessor = FactorPreprocessor()
        df = preprocessor.preprocess_pipeline(
            df, factor_cols,
            winsorize_method='mad',
            standardize_method='zscore'
        )
        print(f"4️⃣ 因子预处理完成")
        
        self.processed_df = df
        
        # 5. 返回结果
        print(f"\n✅ 分析完成!")
        print(f"   最终数据: {len(df)} 行 x {len(df.columns)} 列")
        
        return df


def demo_single_stock_analysis():
    """单股票分析演示"""
    print("=" * 70)
    print("🔬 A股数据 + 量化框架集成演示")
    print("=" * 70)
    
    # 创建适配器
    adapter = AStockDataAdapter()
    
    # 加载数据（只加载平安银行做演示）
    adapter.load_and_transform(stock_code='000001')
    
    # 运行完整因子流程
    df = adapter.run_factor_pipeline('000001')
    
    # 查看结果
    print(f"\n📊 数据预览（最近5天）:")
    print(df[['open', 'high', 'low', 'close', 'volume', 'momentum_20', 'rsi_6']].tail())
    
    print(f"\n📈 因子统计:")
    factor_cols = ['momentum_20', 'rsi_6', 'volatility_20', 'ma_5', 'ma_20']
    print(df[factor_cols].describe().round(4))
    
    # 保存结果
    output_file = './data/000001_factor_analysis.csv'
    df.to_csv(output_file)
    print(f"\n💾 结果已保存: {output_file}")
    
    return adapter, df


def demo_multi_stock_screening():
    """多股票筛选演示"""
    print("\n" + "=" * 70)
    print("🔍 多股票筛选演示")
    print("=" * 70)
    
    adapter = AStockDataAdapter()
    
    # 加载多只股票
    stocks = ['000001', '000002', '600519', '002594', '300750']
    
    results = []
    for code in stocks:
        try:
            adapter.load_and_transform(stock_code=code)
            df = adapter.get_single_stock(code)
            
            # 计算简单指标
            returns_1y = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
            volatility = df['price_change_rate'].std() * np.sqrt(252)
            max_drawdown = ((df['close'] / df['close'].cummax()) - 1).min() * 100
            
            results.append({
                'code': code,
                'name': df['name'].iloc[0] if 'name' in df.columns else '',
                'start_price': df['close'].iloc[0],
                'end_price': df['close'].iloc[-1],
                'return_1y': returns_1y,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe': returns_1y / volatility if volatility > 0 else 0
            })
        except Exception as e:
            print(f"   ⚠️ {code} 分析失败: {e}")
    
    result_df = pd.DataFrame(results)
    print(f"\n📊 多股票对比:")
    print(result_df.to_string(index=False))
    
    return result_df


if __name__ == '__main__':
    # 单股票分析
    adapter, df = demo_single_stock_analysis()
    
    # 多股票筛选
    result_df = demo_multi_stock_screening()
