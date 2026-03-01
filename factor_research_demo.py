"""
完整因子研究流程演示
Day 6-12 综合实战：从原始数据到多因子策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 导入我们创建的模块
from utils.data_fetcher import DataFetcher
from utils.data_processor import DataProcessor, DataStore
from utils.factor_calculator import FactorPipeline
from utils.factor_preprocessor import FactorPreprocessor
from utils.factor_analyzer import FactorAnalyzer
from utils.multi_factor_model import MultiFactorStrategy


def demo_single_stock_factor_research():
    """
    单股票多因子研究演示
    """
    print("=" * 70)
    print("🚀 单股票因子研究完整流程演示")
    print("=" * 70)
    
    # ===== Step 1: 获取数据 =====
    print("\n📌 Step 1: 数据获取")
    fetcher = DataFetcher()
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')
    
    print(f"   获取平安银行(000001) 从 {start_date} 到 {end_date} 的数据...")
    df = fetcher.get_daily_data_ak('000001', start_date, end_date)
    print(f"   获取到 {len(df)} 条日线数据")
    
    # ===== Step 2: 数据预处理 =====
    print("\n📌 Step 2: 数据预处理")
    processor = DataProcessor()
    
    # 清洗数据
    df = processor.clean_price_data(df)
    print("   ✅ 数据清洗完成")
    
    # 计算收益率
    df = processor.calculate_returns(df, periods=[1, 5, 10, 20])
    print("   ✅ 收益率计算完成")
    
    # 保存清洗后的数据
    store = DataStore()
    store.save_stock_data(df, '000001', 'daily_cleaned')
    print("   ✅ 数据已保存到本地")
    
    # ===== Step 3: 因子计算 =====
    print("\n📌 Step 3: 因子计算")
    factor_pipeline = FactorPipeline()
    df = factor_pipeline.calculate_all_factors(df)
    
    # 获取因子列表
    factor_cols = factor_pipeline.get_factor_list(df)
    print(f"   ✅ 共计算 {len(factor_cols)} 个因子")
    print(f"\n   因子示例: {factor_cols[:10]}")
    
    # ===== Step 4: 因子预处理 =====
    print("\n📌 Step 4: 因子预处理")
    preprocessor = FactorPreprocessor()
    
    df = preprocessor.preprocess_pipeline(
        df, factor_cols,
        winsorize_method='mad',
        standardize_method='zscore'
    )
    print("   ✅ 因子预处理完成（去极值+标准化）")
    
    # ===== Step 5: 查看结果 =====
    print("\n📌 Step 5: 预处理结果展示")
    print("\n   原始因子 vs 预处理后因子（最后5日）:")
    
    sample_factors = ['momentum_20', 'rsi_6', 'volatility_20']
    for factor in sample_factors:
        if factor in df.columns:
            print(f"\n   {factor}:")
            print(f"      原始值范围: [{df[factor].min():.4f}, {df[factor].max():.4f}]")
            print(f"      处理后均值: {df[factor].mean():.4f}, 标准差: {df[factor].std():.4f}")
    
    print("\n" + "=" * 70)
    print("✅ 单股票因子研究流程完成！")
    print("=" * 70)
    
    return df, factor_cols


def demo_multi_stock_cross_sectional_analysis():
    """
    多股票截面因子分析演示
    模拟多因子选股策略
    """
    print("\n" + "=" * 70)
    print("🚀 多股票截面因子分析演示")
    print("=" * 70)
    
    # 创建模拟多股票数据（实际应用中应获取真实多股票数据）
    print("\n📌 创建模拟多股票数据...")
    np.random.seed(42)
    
    n_dates = 30
    n_stocks = 100
    dates = pd.date_range(end=datetime.now(), periods=n_dates, freq='B')
    stock_codes = [f'{i:06d}' for i in range(1, n_stocks + 1)]
    
    data = []
    for date in dates:
        for code in stock_codes:
            # 模拟因子数据
            momentum = np.random.randn()
            value = np.random.randn()
            quality = np.random.randn() * 0.5 + momentum * 0.3
            volatility = abs(np.random.randn())
            
            # 模拟未来收益率（与动量因子正相关）
            forward_return = 0.1 * momentum + 0.02 * np.random.randn()
            
            data.append({
                'date': date,
                'code': code,
                'momentum': momentum,
                'value': value,
                'quality': quality,
                'volatility': volatility,
                'market_cap': np.random.lognormal(15, 1),
                'forward_return': forward_return
            })
    
    df = pd.DataFrame(data)
    print(f"   创建完成: {len(df)} 条记录，{n_stocks} 只股票，{n_dates} 个交易日")
    
    # 截面因子预处理
    print("\n📌 截面因子预处理（每月/每日截面处理）")
    preprocessor = FactorPreprocessor()
    factor_cols = ['momentum', 'value', 'quality', 'volatility']
    
    # 对每个截面分别处理
    processed_data = []
    for date, group in df.groupby('date'):
        group = group.copy()
        
        # 截面去极值
        group = preprocessor.winsorize_df(group, factor_cols, method='mad')
        
        # 截面标准化
        group = preprocessor.standardize_df(group, factor_cols, method='zscore')
        
        processed_data.append(group)
    
    df_processed = pd.concat(processed_data, ignore_index=True)
    print("   ✅ 截面预处理完成")
    
    # 因子有效性检验
    print("\n📌 因子有效性检验（IC分析）")
    analyzer = FactorAnalyzer()
    
    ic_results = {}
    for factor in factor_cols:
        ic_series = analyzer.calculate_ic_series(
            df_processed, factor, 'forward_return', 'date'
        )
        ic_stats = analyzer.ic_statistics(ic_series['ic'])
        ic_results[factor] = ic_stats
        
        print(f"\n   {factor}:")
        print(f"      IC均值: {ic_stats['ic_mean']:.4f}")
        print(f"      IC_IR: {ic_stats['ic_ir']:.4f}")
        print(f"      IC>0比例: {ic_stats['ic_ratio_positive']:.2%}")
    
    # 多因子合成
    print("\n📌 多因子合成")
    strategy = MultiFactorStrategy()
    
    # 选择有效因子
    selected = strategy.select_factors(ic_results, min_ic=0.01, min_ir=0.05)
    
    if len(selected) >= 2:
        # IC加权合成
        ic_values = {f: ic_results[f]['ic_mean'] for f in selected}
        composite = strategy.build_composite_factor(
            df_processed, selected, method='ic', ic_values=ic_values
        )
        df_processed['composite_factor'] = composite
        
        # 评估复合因子
        composite_ic_series = analyzer.calculate_ic_series(
            df_processed, 'composite_factor', 'forward_return', 'date'
        )
        composite_stats = analyzer.ic_statistics(composite_ic_series['ic'])
        
        print(f"\n   复合因子效果:")
        print(f"      IC均值: {composite_stats['ic_mean']:.4f}")
        print(f"      IC_IR: {composite_stats['ic_ir']:.4f}")
    
    # 分层回测
    print("\n📌 分层回测")
    quantile_returns, cumulative = analyzer.quantile_backtest(
        df_processed, 'momentum', 'forward_return', n_quantiles=5, date_col='date'
    )
    
    print("\n   各层平均收益:")
    mean_returns = quantile_returns.mean()
    for q in mean_returns.index:
        print(f"      {q}: {mean_returns[q]:.4f}")
    
    # 多空收益
    spread = analyzer.calculate_spread(quantile_returns)
    if spread is not None:
        print(f"\n   多空组合(Q5-Q1):")
        print(f"      平均收益: {spread.mean():.4f}")
        print(f"      胜率: {(spread > 0).mean():.2%}")
    
    print("\n" + "=" * 70)
    print("✅ 多股票截面分析完成！")
    print("=" * 70)
    
    return df_processed, ic_results


def print_learning_summary():
    """打印学习总结"""
    print("\n" + "=" * 70)
    print("📚 Day 6-12 学习总结：数据与因子工程")
    print("=" * 70)
    
    summary = """
✅ 已完成内容:

1. 数据获取与预处理 (Day 6-7)
   • utils/data_fetcher.py - AKShare数据接口
   • utils/data_processor.py - 数据清洗、复权、收益率计算
   • utils/data_fetcher.py - 自动故障切换到模拟数据

2. 因子计算 (Day 8)
   • utils/factor_calculator.py
     - 技术指标：MA、RSI、MACD、布林带、波动率、成交量、动量
     - 财务因子：估值、质量指标（需配合财务数据）
     - FactorPipeline: 一键计算所有因子

3. 因子预处理 (Day 9)
   • utils/factor_preprocessor.py
     - 去极值：MAD、标准差、分位数方法
     - 标准化：Z-Score、Rank、Min-Max
     - 中性化：市值中性化、行业中性化
     - 完整预处理流水线

4. 因子有效性检验 (Day 10)
   • utils/factor_analyzer.py
     - IC分析：IC均值、IC_IR、IC胜率
     - 分层回测：多空组合、单调性检验
     - IC衰减：因子预测能力持续时间

5. 多因子合成与策略 (Day 11-12)
   • utils/multi_factor_model.py
     - 合成方法：等权、IC加权、IR加权、ML加权、PCA
     - 多因子策略：因子筛选、复合因子构建、选股
     - 完整研究流水线：FactorResearchPipeline

🔧 关键工具函数:
   • DataProcessor.clean_price_data() - 清洗价格数据
   • FactorPreprocessor.preprocess_pipeline() - 预处理流水线
   • FactorAnalyzer.generate_factor_report() - 生成因子报告
   • MultiFactorStrategy.select_factors() - 因子筛选
   • MultiFactorStrategy.build_composite_factor() - 因子合成

📝 使用示例:
   python factor_research_demo.py

🚀 下一步建议:
   1. 在Jupyter Lab中交互式探索因子
   2. 接入真实A股全市场数据进行回测
   3. 尝试机器学习因子合成
   4. 进入Day 13-20：策略开发与回测框架
"""
    print(summary)


if __name__ == '__main__':
    # 运行演示
    df_single, factors = demo_single_stock_factor_research()
    df_multi, ic_results = demo_multi_stock_cross_sectional_analysis()
    print_learning_summary()
