"""
持仓盈亏分析 - 验证回测收益与交易盈亏的差异
"""

import pandas as pd
from typing import Dict

def analyze_position_pnl(result: Dict, initial_cash: float = 1000000):
    """
    分析持仓盈亏 vs 已实现盈亏
    
    Parameters
    ----------
    result : dict
        回测结果字典
    initial_cash : float
        初始资金
    """
    trade_log = result.get('trade_log', [])
    final_value = result.get('final_value', 0)
    
    # 1. 计算已实现盈亏（已平仓交易）
    trades_df = pd.DataFrame(trade_log)
    if trades_df.empty:
        print("⚠️ 没有交易记录")
        return
    
    # 统计每只股票的累计买入和卖出
    stock_stats = {}
    for _, trade in trades_df.iterrows():
        code = trade['code']
        if code not in stock_stats:
            stock_stats[code] = {'buy_value': 0, 'sell_value': 0, 'buy_size': 0, 'sell_size': 0}
        
        if trade['type'] == 'buy':
            stock_stats[code]['buy_value'] += trade['value']
            stock_stats[code]['buy_size'] += trade['size']
        else:
            stock_stats[code]['sell_value'] += abs(trade['value'])
            stock_stats[code]['sell_size'] += trade['size']
    
    # 2. 分离已完成和持仓中的股票
    completed_pnl = 0  # 已实现盈亏
    position_cost = 0  # 持仓成本
    
    for code, stats in stock_stats.items():
        if stats['sell_size'] >= stats['buy_size']:
            # 已清仓（或部分清仓，简化计算）
            completed_pnl += stats['sell_value'] - stats['buy_value']
        else:
            # 仍持有
            avg_cost = stats['buy_value'] / stats['buy_size'] if stats['buy_size'] > 0 else 0
            sold_cost = avg_cost * stats['sell_size']
            position_cost += (stats['buy_value'] - sold_cost)
            
            # 已卖出部分的盈亏
            if stats['sell_size'] > 0:
                sold_profit = stats['sell_value'] - sold_cost
                completed_pnl += sold_profit
    
    # 3. 计算持仓市值
    position_value = final_value - (initial_cash + completed_pnl)  # 近似估算
    position_pnl = position_value - position_cost
    
    print("=" * 70)
    print("📊 收益分解分析")
    print("=" * 70)
    print(f"\n初始资金: {initial_cash:,.2f}")
    print(f"最终资金: {final_value:,.2f}")
    print(f"总收益: {final_value - initial_cash:,.2f} ({(final_value/initial_cash-1)*100:.2f}%)")
    
    print(f"\n【已实现盈亏】(已卖出股票)")
    print(f"   盈亏: {completed_pnl:,.2f}")
    
    print(f"\n【持仓盈亏】(仍持有的股票)")
    print(f"   成本: {position_cost:,.2f}")
    print(f"   市值: {position_value:,.2f}")
    print(f"   浮盈: {position_pnl:,.2f}")
    
    print(f"\n【验证】")
    print(f"   已实现 + 持仓浮盈 = {completed_pnl + position_pnl:,.2f}")
    print(f"   实际总收益 = {final_value - initial_cash:,.2f}")
    print(f"   差异 = {(completed_pnl + position_pnl) - (final_value - initial_cash):,.2f}")
    
    # 4. 持仓股票列表
    print(f"\n【期末持仓股票】")
    position_stocks = []
    for code, stats in stock_stats.items():
        if stats['buy_size'] > stats['sell_size']:
            hold_size = stats['buy_size'] - stats['sell_size']
            avg_cost = stats['buy_value'] / stats['buy_size']
            position_stocks.append({
                'code': code,
                'size': hold_size,
                'cost': avg_cost * hold_size,
                'avg_price': avg_cost
            })
    
    if position_stocks:
        df = pd.DataFrame(position_stocks)
        print(df.to_string(index=False))
    else:
        print("   (无持仓)")


# 使用示例
if __name__ == "__main__":
    # 假设 result 是回测结果
    # analyze_position_pnl(result, initial_cash=1000000)
    pass
