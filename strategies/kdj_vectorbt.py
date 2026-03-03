"""
KDJ 策略 - VectorBT 向量化回测版本
优势: 速度快，适合参数优化
"""

import numpy as np
import pandas as pd
import vectorbt as vbt


def calculate_kdj(close, high, low, n=9, m1=3, m2=3):
    """
    计算KDJ指标
    
    Parameters:
    -----------
    close, high, low : pd.Series
        价格序列
    n, m1, m2 : int
        KDJ参数
    
    Returns:
    --------
    k, d, j : pd.Series
    """
    # RSV
    low_n = low.rolling(window=n, min_periods=n).min()
    high_n = high.rolling(window=n, min_periods=n).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    
    # K值
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    
    # D值
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    
    # J值
    j = 3 * k - 2 * d
    
    return k, d, j


def kdj_strategy_signals(df, j_threshold=-5, take_profit=0.20, use_stop_loss=False, stop_loss=0.10):
    """
    生成KDJ策略交易信号
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含OHLCV的数据
    j_threshold : float
        J值买入阈值
    take_profit : float
        止盈比例
    use_stop_loss : bool
        是否使用止损
    stop_loss : float
        止损比例
    
    Returns:
    --------
    entries, exits : pd.Series
        买入/卖出信号（bool）
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # 计算KDJ
    k, d, j = calculate_kdj(close, high, low)
    
    # 买入信号: J < -5
    entries = j < j_threshold
    
    # 初始卖出信号为False
    exits = pd.Series(False, index=df.index)
    
    # 止盈逻辑（向量化实现）
    # 找到所有买入点
    entry_points = entries[entries].index
    
    for entry_date in entry_points:
        entry_price = close.loc[entry_date]
        
        # 计算从买入点之后的收益率
        future_returns = (close.loc[entry_date:] / entry_price - 1)
        
        # 找到第一个达到止盈或止损的日期
        take_profit_date = future_returns[future_returns >= take_profit].index
        
        if use_stop_loss:
            stop_loss_date = future_returns[future_returns <= -stop_loss].index
        else:
            stop_loss_date = pd.DatetimeIndex([])
        
        # 取最早的退出日期
        exit_dates = list(take_profit_date) + list(stop_loss_date)
        if exit_dates:
            first_exit = min(exit_dates)
            if first_exit in exits.index:
                exits.loc[first_exit] = True
    
    # 确保没有同时出现买入和卖出信号
    exits = exits & ~entries
    
    return entries, exits


def run_kdj_backtest(df, init_cash=100000, fees=0.001, **kwargs):
    """
    运行KDJ策略回测
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据
    init_cash : float
        初始资金
    fees : float
        手续费
    **kwargs : 
        传递给kdj_strategy_signals的参数
    
    Returns:
    --------
    portfolio : vbt.Portfolio
    """
    close = df['close']
    
    # 生成信号
    entries, exits = kdj_strategy_signals(df, **kwargs)
    
    # 运行回测
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
        freq='1d'
    )
    
    return portfolio


def kdj_parameter_optimization(df, j_range=range(-20, 0, 2), tp_range=[0.10, 0.15, 0.20, 0.25, 0.30]):
    """
    KDJ参数优化
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据
    j_range : range
        J值阈值范围
    tp_range : list
        止盈比例范围
    
    Returns:
    --------
    best_params : dict
        最优参数
    results : pd.DataFrame
        所有参数结果
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # 计算KDJ
    k, d, j = calculate_kdj(close, high, low)
    
    results = []
    
    for j_th in j_range:
        for tp in tp_range:
            # 生成信号
            entries = j < j_th
            
            # 简化的回测（只用vectorbt的固定退出）
            portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=pd.Series(False, index=close.index),
                sl_stop=tp if tp < 1 else None,  # 简化处理
                init_cash=100000,
                fees=0.001,
                freq='1d'
            )
            
            results.append({
                'j_threshold': j_th,
                'take_profit': tp,
                'total_return': portfolio.total_return(),
                'sharpe': portfolio.sharpe_ratio(),
                'max_drawdown': portfolio.max_drawdown(),
                'trades': portfolio.trades.count()
            })
    
    results_df = pd.DataFrame(results)
    
    # 找最优参数（按夏普比率）
    best_idx = results_df['sharpe'].idxmax()
    best_params = results_df.loc[best_idx].to_dict()
    
    return best_params, results_df


if __name__ == '__main__':
    print("KDJ VectorBT 策略模块")
    print("使用示例:")
    print("  from strategies.kdj_vectorbt import run_kdj_backtest")
    print("  portfolio = run_kdj_backtest(df, j_threshold=-5, take_profit=0.20)")
    print("  print(portfolio.stats())")
