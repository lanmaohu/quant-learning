#!/usr/bin/env python
"""
KDJ策略回测主程序
支持 Backtrader 和 VectorBT 两种回测引擎

策略逻辑:
- 买入: KDJ的J值 < -5（超卖）
- 止盈: 持仓收益 >= 20%
- 可选: 止损
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_fetcher import DataFetcher
from utils.data_processor import DataProcessor
from strategies.kdj_strategy import KDJStrategy


def run_backtrader_backtest(symbol='000001', start_date='20220101', end_date='20241231',
                            j_threshold=-5, take_profit=0.20, print_log=True):
    """
    使用 Backtrader 运行KDJ策略回测
    """
    import backtrader as bt
    
    print("=" * 70)
    print("🚀 KDJ策略回测 - Backtrader")
    print("=" * 70)
    print(f"\n策略参数:")
    print(f"   买入条件: J值 < {j_threshold}")
    print(f"   止盈条件: 收益 >= {take_profit*100:.0f}%")
    print(f"\n股票: {symbol}")
    print(f"回测区间: {start_date} ~ {end_date}")
    
    # 1. 获取数据
    print("\n📥 获取数据...")
    fetcher = DataFetcher()
    df = fetcher.get_daily_data_ak(symbol, start_date, end_date)
    print(f"   获取到 {len(df)} 条数据")
    
    # 2. 准备 Backtrader 数据
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    
    # 3. 初始化 Cerebro
    cerebro = bt.Cerebro()
    
    # 添加数据
    cerebro.adddata(data, name=symbol)
    
    # 添加策略
    cerebro.addstrategy(
        KDJStrategy,
        j_threshold=j_threshold,
        take_profit=take_profit,
        print_log=print_log
    )
    
    # 设置初始资金
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)  # 千分之一手续费
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 4. 运行回测
    print("\n💰 初始资金: 100,000.00")
    print("-" * 70)
    
    results = cerebro.run()
    strat = results[0]
    
    print("-" * 70)
    print(f"💰 最终资金: {cerebro.broker.getvalue():,.2f}")
    
    # 5. 打印分析结果
    print("\n" + "=" * 70)
    print("📊 回测结果")
    print("=" * 70)
    
    # 收益率
    returns = strat.analyzers.returns.get_analysis()
    print(f"总收益率: {returns['rtot']*100:.2f}%")
    print(f"年化收益率: {returns['rnorm']*100:.2f}%")
    
    # 夏普比率
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"夏普比率: {sharpe.get('sharperatio', 0):.3f}")
    
    # 最大回撤
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"最大回撤: {drawdown['max']['drawdown']:.2f}%")
    
    # 交易统计
    trades = strat.analyzers.trades.get_analysis()
    if trades.get('total', {}).get('total', 0) > 0:
        print(f"\n交易统计:")
        print(f"   总交易次数: {trades['total']['total']}")
        print(f"   盈利次数: {trades.get('won', {}).get('total', 0)}")
        print(f"   亏损次数: {trades.get('lost', {}).get('total', 0)}")
        
        won_total = trades.get('won', {}).get('pnl', {}).get('total', 0)
        lost_total = trades.get('lost', {}).get('pnl', {}).get('total', 0)
        print(f"   总盈利: {won_total:,.2f}")
        print(f"   总亏损: {abs(lost_total):,.2f}")
        if lost_total != 0:
            print(f"   盈亏比: {abs(won_total/lost_total):.2f}")
    
    # 6. 绘图（可选）
    # cerebro.plot(style='candlestick')
    
    return strat


def run_vectorbt_backtest(symbol='000001', start_date='20220101', end_date='20241231',
                          j_threshold=-5, take_profit=0.20):
    """
    使用 VectorBT 运行KDJ策略回测（向量化，速度快）
    """
    print("=" * 70)
    print("🚀 KDJ策略回测 - VectorBT")
    print("=" * 70)
    print(f"\n策略参数:")
    print(f"   买入条件: J值 < {j_threshold}")
    print(f"   止盈条件: 收益 >= {take_profit*100:.0f}%")
    
    # 导入VectorBT策略
    from strategies.kdj_vectorbt import run_kdj_backtest
    
    # 1. 获取数据
    print(f"\n📥 获取 {symbol} 数据...")
    fetcher = DataFetcher()
    df = fetcher.get_daily_data_ak(symbol, start_date, end_date)
    print(f"   获取到 {len(df)} 条数据")
    
    # 2. 运行回测
    print("\n🔄 运行回测...")
    portfolio = run_kdj_backtest(
        df,
        j_threshold=j_threshold,
        take_profit=take_profit,
        init_cash=100000,
        fees=0.001
    )
    
    # 3. 打印结果
    print("\n" + "=" * 70)
    print("📊 回测结果")
    print("=" * 70)
    
    stats = portfolio.stats()
    print(f"总收益率: {portfolio.total_return()*100:.2f}%")
    print(f"年化收益率: {portfolio.annualized_return()*100:.2f}%")
    print(f"夏普比率: {portfolio.sharpe_ratio():.3f}")
    print(f"最大回撤: {portfolio.max_drawdown()*100:.2f}%")
    print(f"交易次数: {portfolio.trades.count()}")
    print(f"胜率: {portfolio.trades.win_rate()*100:.1f}%")
    
    # 4. 绘制图表
    try:
        fig = portfolio.plot()
        fig.write_html(f"./data/kdj_backtest_{symbol}.html")
        print(f"\n✅ 图表已保存: ./data/kdj_backtest_{symbol}.html")
    except:
        pass
    
    return portfolio


def compare_strategies(symbol='000001'):
    """
    对比不同参数的KDJ策略
    """
    print("=" * 70)
    print("🔍 KDJ策略参数对比")
    print("=" * 70)
    
    # 参数组合
    params_list = [
        {'j_threshold': -5, 'take_profit': 0.15},
        {'j_threshold': -5, 'take_profit': 0.20},
        {'j_threshold': -5, 'take_profit': 0.25},
        {'j_threshold': -10, 'take_profit': 0.20},
        {'j_threshold': -15, 'take_profit': 0.20},
    ]
    
    results = []
    
    for params in params_list:
        print(f"\n测试参数: J<{params['j_threshold']}, 止盈{params['take_profit']*100:.0f}%")
        
        try:
            portfolio = run_vectorbt_backtest(
                symbol=symbol,
                j_threshold=params['j_threshold'],
                take_profit=params['take_profit']
            )
            
            results.append({
                'j_threshold': params['j_threshold'],
                'take_profit': params['take_profit'],
                'total_return': portfolio.total_return(),
                'sharpe': portfolio.sharpe_ratio(),
                'max_drawdown': portfolio.max_drawdown(),
                'trades': portfolio.trades.count()
            })
        except Exception as e:
            print(f"   错误: {e}")
    
    # 对比结果
    print("\n" + "=" * 70)
    print("📊 参数对比结果")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 找出最优
    best_idx = results_df['sharpe'].idxmax()
    best = results_df.loc[best_idx]
    print(f"\n✅ 最优参数:")
    print(f"   J阈值: {best['j_threshold']}")
    print(f"   止盈: {best['take_profit']*100:.0f}%")
    print(f"   夏普比率: {best['sharpe']:.3f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='KDJ策略回测')
    parser.add_argument('--engine', type=str, default='vectorbt', 
                       choices=['backtrader', 'vectorbt', 'compare'],
                       help='回测引擎')
    parser.add_argument('--symbol', type=str, default='000001', help='股票代码')
    parser.add_argument('--start', type=str, default='20220101', help='开始日期')
    parser.add_argument('--end', type=str, default='20241231', help='结束日期')
    parser.add_argument('--j', type=float, default=-5, help='J值阈值')
    parser.add_argument('--tp', type=float, default=0.20, help='止盈比例')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    
    args = parser.parse_args()
    
    if args.engine == 'backtrader':
        run_backtrader_backtest(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            j_threshold=args.j,
            take_profit=args.tp,
            print_log=not args.quiet
        )
    elif args.engine == 'vectorbt':
        run_vectorbt_backtest(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            j_threshold=args.j,
            take_profit=args.tp
        )
    elif args.engine == 'compare':
        compare_strategies(symbol=args.symbol)


if __name__ == '__main__':
    # 默认运行 VectorBT 回测
    if len(sys.argv) == 1:
        run_vectorbt_backtest('000001', '20220101', '20241231', -5, 0.20)
    else:
        main()
