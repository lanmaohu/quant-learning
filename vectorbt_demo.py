"""
VectorBT 入门示例
向量化回测，速度极快，适合参数优化和ML策略
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.data_fetcher import DataFetcher
from utils.constants import DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE


def run_vectorbt_backtest(symbol='000001', start_date='20220101', end_date='20241231'):
    """
    使用 VectorBT 运行双均线策略回测
    """
    print("=" * 60)
    print("🚀 VectorBT 双均线策略回测")
    print("=" * 60)
    
    # 1. 获取数据
    print(f"\n📥 获取 {symbol} 数据...")
    fetcher = DataFetcher()
    df = fetcher.get_daily_data_ak(symbol, start_date, end_date)
    
    # 提取收盘价
    close = df['close']
    
    # 2. 计算指标
    print("\n📊 计算均线指标...")
    short_ma = vbt.MA.run(close, window=20)
    long_ma = vbt.MA.run(close, window=60)
    
    # 3. 生成信号
    # 短期均线上穿长期均线 -> 买入 (1)
    # 短期均线下穿长期均线 -> 卖出 (-1)
    entries = short_ma.ma_crossed_above(long_ma)
    exits = short_ma.ma_crossed_below(long_ma)
    
    print(f"买入信号次数: {entries.sum()}")
    print(f"卖出信号次数: {exits.sum()}")
    
    # 4. 运行回测
    print("\n🔄 运行回测...")
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=DEFAULT_COMMISSION_RATE,  # 手续费
        slippage=DEFAULT_SLIPPAGE,      # 滑点
        freq='1d'        # 日线数据
    )
    
    # 5. 查看结果
    print("\n" + "=" * 60)
    print("📈 回测结果")
    print("=" * 60)
    
    print(f"总收益率: {portfolio.total_return() * 100:.2f}%")
    print(f"年化收益率: {portfolio.annualized_return() * 100:.2f}%")
    print(f"夏普比率: {portfolio.sharpe_ratio():.3f}")
    print(f"最大回撤: {portfolio.max_drawdown() * 100:.2f}%")
    print(f"胜率: {portfolio.trades.win_rate() * 100:.1f}%")
    print(f"交易次数: {portfolio.trades.count()}")
    
    # 6. 绘制图表
    fig = portfolio.plot()
    fig.write_html(f"./data/vectorbt_backtest_{symbol}.html")
    print(f"\n✅ 图表已保存: ./data/vectorbt_backtest_{symbol}.html")
    
    return portfolio


def parameter_optimization(symbol='000001'):
    """
    VectorBT 参数优化示例 - 寻找最佳均线组合
    """
    print("\n" + "=" * 60)
    print("🔍 VectorBT 参数优化示例")
    print("=" * 60)
    
    # 获取数据
    fetcher = DataFetcher()
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')
    df = fetcher.get_daily_data_ak(symbol, start_date, end_date)
    close = df['close']
    
    # 定义参数范围
    short_windows = range(5, 50, 5)   # 5, 10, 15, ..., 45
    long_windows = range(20, 100, 10)  # 20, 30, 40, ..., 90
    
    print(f"测试 {len(short_windows)} × {len(long_windows)} = {len(short_windows) * len(long_windows)} 种参数组合...")
    
    # 使用向量化计算所有组合
    short_ma = vbt.MA.run(close, window=list(short_windows), short_name='short')
    long_ma = vbt.MA.run(close, window=list(long_windows), short_name='long')
    
    # 建立参数网格
    entries = short_ma.ma_crossed_above(long_ma)
    exits = short_ma.ma_crossed_below(long_ma)
    
    # 运行回测
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=DEFAULT_COMMISSION_RATE,
        freq='1d'
    )
    
    # 获取收益热力图
    returns = portfolio.annualized_return()
    
    # 找到最佳参数
    best_idx = returns.values.argmax()
    best_return = returns.values.flat[best_idx]
    best_params = returns.index[best_idx]
    
    print(f"\n🏆 最佳参数组合:")
    print(f"   短期均线: {best_params[0]} 日")
    print(f"   长期均线: {best_params[1]} 日")
    print(f"   年化收益: {best_return * 100:.2f}%")
    
    # 绘制热力图
    fig = returns.vbt.heatmap(
        x_level='short_window',
        y_level='long_window',
        title='年化收益率热力图 (参数优化)',
        colorscale='RdYlGn'
    )
    fig.write_html("./data/parameter_optimization_heatmap.html")
    print(f"✅ 热力图已保存: ./data/parameter_optimization_heatmap.html")
    
    return portfolio


if __name__ == '__main__':
    # 1. 基础回测
    portfolio = run_vectorbt_backtest('000001', '20220101', '20241231')
    
    # 2. 参数优化（可选，取消注释运行）
    # parameter_optimization('000001')
