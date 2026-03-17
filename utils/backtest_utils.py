"""
回测公共工具函数

统一提供两个策略文件（threshold / topk）共用的:
- create_strategy_chart(): 收益曲线 + 买卖点图表
- analyze_trade_profit_loss(): 交易盈亏分析
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict


def create_strategy_chart(result: Dict, title: str = "Strategy", save_path: str = None):
    """
    创建回测结果图表（收益曲线 + 买卖点标记）

    兼容 Threshold 策略和 TopK 策略的结果字典：
    - Threshold: result['portfolio_values'] 列表，每项含 date/portfolio_value/cash
    - TopK:      result['daily_portfolio'] 列表，每项含 date/portfolio_value/cash/n_positions/rebalanced

    Parameters
    ----------
    result : dict
        回测结果字典（来自 run_threshold_backtest_ultra 或 run_topk_backtest_ultra_optimized）
    title : str
        图表标题
    save_path : str, optional
        保存路径（None 则不保存）
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # 兼容两种 key
    raw = result.get('daily_portfolio') or result.get('portfolio_values', [])
    if not raw:
        print("⚠️ 没有组合价值数据，无法绘制图表")
        return

    portfolio_df = pd.DataFrame(raw)
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df.set_index('date', inplace=True)

    initial_value = result['initial_cash']
    portfolio_df['cum_return'] = (portfolio_df['portfolio_value'] / initial_value - 1) * 100
    portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change() * 100

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

    # ===== 子图1: 累计收益曲线 + 买卖点 =====
    ax1 = axes[0]
    ax1.plot(portfolio_df.index, portfolio_df['cum_return'],
             label='Cumulative Return (%)', color='steelblue', linewidth=2)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(portfolio_df.index, 0, portfolio_df['cum_return'],
                     where=portfolio_df['cum_return'] >= 0, alpha=0.3, color='green')
    ax1.fill_between(portfolio_df.index, 0, portfolio_df['cum_return'],
                     where=portfolio_df['cum_return'] < 0, alpha=0.3, color='red')

    trade_log = result.get('trade_log', [])
    buy_trades = [t for t in trade_log if t.get('type') == 'buy']
    sell_trades = [t for t in trade_log if t.get('type') == 'sell']

    for trade in buy_trades:
        date = pd.to_datetime(trade['date'])
        if date in portfolio_df.index:
            ax1.scatter(date, portfolio_df.loc[date, 'cum_return'],
                        marker='^', color='red', s=100, zorder=5,
                        label='Buy' if trade == buy_trades[0] else "")

    for trade in sell_trades:
        date = pd.to_datetime(trade['date'])
        if date in portfolio_df.index:
            ax1.scatter(date, portfolio_df.loc[date, 'cum_return'],
                        marker='v', color='green', s=100, zorder=5,
                        label='Sell' if trade == sell_trades[0] else "")

    # 再平衡标记（TopK 策略特有）
    if 'rebalanced' in portfolio_df.columns:
        rebalance_df = portfolio_df[portfolio_df['rebalanced'] == True]
        if len(rebalance_df) > 0:
            ax1.scatter(rebalance_df.index,
                        portfolio_df.loc[rebalance_df.index, 'cum_return'],
                        marker='|', color='purple', s=50, alpha=0.5, label='Rebalance')

    stats_text = (
        f"Total Return: {result['total_return']*100:.2f}%\n"
        f"Annual Return: {result.get('annual_return', 0)*100:.2f}%\n"
        f"Sharpe Ratio: {result['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {result['max_drawdown']*100:.2f}%\n"
        f"Trades: {len(trade_log)}"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax1.set_title(f'{title} - Return Curve & Trade Points', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ===== 子图2: 每日收益柱状图 =====
    ax2 = axes[1]
    colors = ['green' if r >= 0 else 'red' for r in portfolio_df['daily_return']]
    ax2.bar(portfolio_df.index, portfolio_df['daily_return'], color=colors, alpha=0.6)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Daily Return (%)', fontsize=11)
    ax2.set_title('Daily Returns', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # ===== 子图3: 持仓数量（TopK）或现金比例（Threshold）=====
    ax3 = axes[2]
    portfolio_df['cash_ratio'] = portfolio_df['cash'] / portfolio_df['portfolio_value'] * 100

    if 'n_positions' in portfolio_df.columns:
        # TopK 策略：双轴，左轴持仓数，右轴现金比例
        ax3_twin = ax3.twinx()
        ax3.plot(portfolio_df.index, portfolio_df['n_positions'],
                 color='blue', linewidth=1.5, label='Positions')
        ax3_twin.fill_between(portfolio_df.index, 0, portfolio_df['cash_ratio'],
                              alpha=0.3, color='orange', label='Cash Ratio')
        ax3.set_ylabel('Number of Positions', fontsize=11, color='blue')
        ax3_twin.set_ylabel('Cash Ratio (%)', fontsize=11, color='orange')
        ax3.set_title('Positions & Cash', fontsize=12)
    else:
        # Threshold 策略：仅现金比例
        ax3.plot(portfolio_df.index, portfolio_df['cash_ratio'],
                 color='orange', linewidth=1.5, label='Cash Ratio (%)')
        ax3.fill_between(portfolio_df.index, 0, portfolio_df['cash_ratio'],
                         alpha=0.3, color='orange')
        ax3.set_ylabel('Cash Ratio (%)', fontsize=11)
        ax3.set_title('Cash Position', fontsize=12)
        ax3.legend(loc='upper right')

    ax3.set_xlabel('Date', fontsize=11)
    ax3.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 图表已保存: {save_path}")

    plt.show()


def analyze_trade_profit_loss(result: Dict, save_path: str = None):
    """
    分析所有交易的盈亏情况（FIFO 匹配买卖记录）

    Parameters
    ----------
    result : dict
        回测结果字典（包含 trade_log）
    save_path : str, optional
        保存图表路径（None 则只打印统计，不绘图）

    Returns
    -------
    analysis_df : pd.DataFrame or None
        每笔完成交易的盈亏明细
    """
    import matplotlib.pyplot as plt

    trade_log = result.get('trade_log', [])
    if not trade_log:
        print("⚠️ 没有交易记录")
        return None

    trades_df = pd.DataFrame(trade_log)
    trades_df['date'] = pd.to_datetime(trades_df['date'])

    buy_trades = trades_df[trades_df['type'] == 'buy'].copy()
    sell_trades = trades_df[trades_df['type'] == 'sell'].copy()

    completed_trades = []
    for code in trades_df['code'].unique():
        code_buys = buy_trades[buy_trades['code'] == code].sort_values('date')
        code_sells = sell_trades[sell_trades['code'] == code].sort_values('date')

        for _, sell in code_sells.iterrows():
            matching_buys = code_buys[code_buys['date'] < sell['date']]
            if len(matching_buys) > 0:
                buy = matching_buys.iloc[-1]
                profit = sell['size'] * sell['price'] - buy['size'] * buy['price']
                profit_pct = (sell['price'] - buy['price']) / buy['price'] * 100
                holding_days = (sell['date'] - buy['date']).days

                completed_trades.append({
                    'code': code,
                    'buy_date': buy['date'],
                    'sell_date': sell['date'],
                    'buy_price': buy['price'],
                    'sell_price': sell['price'],
                    'size': sell['size'],
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'holding_days': holding_days
                })
                code_buys = code_buys.drop(buy.name)

    if not completed_trades:
        print("⚠️ 没有完成的交易对（需要有买入和卖出记录）")
        return None

    analysis_df = pd.DataFrame(completed_trades)

    # 统计信息
    total_trades = len(analysis_df)
    winning_trades = len(analysis_df[analysis_df['profit'] > 0])
    losing_trades = len(analysis_df[analysis_df['profit'] < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    print("\n" + "=" * 70)
    print("📊 交易盈亏分析")
    print("=" * 70)
    print(f"\n总交易次数: {total_trades}")
    print(f"盈利次数: {winning_trades} ({win_rate:.1f}%)")
    print(f"亏损次数: {losing_trades} ({100-win_rate:.1f}%)")
    print(f"\n总盈亏: {analysis_df['profit'].sum():,.2f}")
    print(f"平均盈亏: {analysis_df['profit'].mean():,.2f}")
    print(f"平均收益率: {analysis_df['profit_pct'].mean():.2f}%")

    if winning_trades > 0:
        print(f"\n平均盈利: {analysis_df[analysis_df['profit'] > 0]['profit'].mean():,.2f}")
        print(f"最大盈利: {analysis_df[analysis_df['profit'] > 0]['profit'].max():,.2f}")

    if losing_trades > 0:
        print(f"平均亏损: {analysis_df[analysis_df['profit'] < 0]['profit'].mean():,.2f}")
        print(f"最大亏损: {analysis_df[analysis_df['profit'] < 0]['profit'].min():,.2f}")

    print(f"\n平均持仓天数: {analysis_df['holding_days'].mean():.1f}天")

    # 交易明细
    print("\n" + "-" * 70)
    print("📋 交易明细（按盈亏排序）")
    print("-" * 70)
    print(f"{'排名':<4} {'代码':<10} {'买入价':<10} {'卖出价':<10} {'股数':<10} {'收益率':<10} {'盈亏':<12} {'持仓天数':<8}")
    print("-" * 90)
    for i, (_, row) in enumerate(analysis_df.sort_values('profit', ascending=False).iterrows(), 1):
        print(f"{i:<4} {row['code']:<10} "
              f"{row['buy_price']:<10.2f} "
              f"{row['sell_price']:<10.2f} "
              f"{int(row['size']):<10} "
              f"{row['profit_pct']:>+8.2f}% "
              f"{row['profit']:>+12.2f} {int(row['holding_days']):<8}")
    print("=" * 70)

    # 可视化（仅在 save_path 非 None 时绘制）
    if save_path is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 盈亏分布
        ax1 = axes[0, 0]
        sorted_pct = analysis_df['profit_pct'].sort_values(ascending=True)
        colors = ['green' if p > 0 else 'red' for p in sorted_pct]
        ax1.bar(range(len(analysis_df)), sorted_pct, color=colors)
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Trade')
        ax1.set_ylabel('Profit %')
        ax1.set_title('Trade P&L Distribution (Sorted)')
        ax1.grid(True, alpha=0.3)

        # 2. 盈亏散点图（按时间）
        ax2 = axes[0, 1]
        scatter_colors = ['green' if p > 0 else 'red' for p in analysis_df['profit']]
        ax2.scatter(analysis_df['buy_date'], analysis_df['profit_pct'],
                    c=scatter_colors, alpha=0.6, s=50)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Buy Date')
        ax2.set_ylabel('Profit %')
        ax2.set_title('Trade P&L Over Time')
        ax2.grid(True, alpha=0.3)

        # 3. 累计盈亏曲线
        ax3 = axes[1, 0]
        by_date = analysis_df.sort_values('sell_date').copy()
        by_date['cum_profit'] = by_date['profit'].cumsum()
        ax3.plot(by_date['sell_date'], by_date['cum_profit'], linewidth=2, color='steelblue')
        ax3.fill_between(by_date['sell_date'], 0, by_date['cum_profit'],
                         where=by_date['cum_profit'] >= 0, alpha=0.3, color='green')
        ax3.fill_between(by_date['sell_date'], 0, by_date['cum_profit'],
                         where=by_date['cum_profit'] < 0, alpha=0.3, color='red')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative Profit')
        ax3.set_title('Cumulative Trade P&L')
        ax3.grid(True, alpha=0.3)

        # 4. 盈亏饼图
        ax4 = axes[1, 1]
        ax4.pie([winning_trades, losing_trades], explode=(0.05, 0),
                labels=['Win', 'Loss'], colors=['#66b3ff', '#ff9999'],
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax4.set_title(f'Win Rate ({win_rate:.1f}%)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 盈亏分析图表已保存: {save_path}")
        plt.show()

    return analysis_df


def setup_analyzers(cerebro: bt.Cerebro) -> None:
    """注册标准回测分析器（SharpeRatio / DrawDown / Returns / TradeAnalyzer）"""
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')


def create_result_dict(strat, cerebro: bt.Cerebro, strategy_name: str,
                       initial_cash: float, elapsed_time: float = 0,
                       portfolio_key: str = 'portfolio_values') -> Dict:
    """
    从 backtrader 策略对象提取标准化结果字典

    Parameters
    ----------
    strat : bt.Strategy
        已运行的策略实例
    cerebro : bt.Cerebro
        已运行的 Cerebro 实例
    strategy_name : str
        策略名称（写入结果字典）
    initial_cash : float
        初始资金
    elapsed_time : float
        回测耗时（秒）
    portfolio_key : str
        组合价值记录在 strat 上的属性名（'portfolio_values' 或 'daily_portfolio'）

    Returns
    -------
    dict
        包含标准回测指标的字典
    """
    final_value = cerebro.broker.getvalue()
    returns_a = strat.analyzers.returns.get_analysis()
    sharpe_a = strat.analyzers.sharpe.get_analysis()
    drawdown_a = strat.analyzers.drawdown.get_analysis()
    trades_a = strat.analyzers.trades.get_analysis()

    result = {
        'strategy': strategy_name,
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': (final_value / initial_cash - 1),
        'annual_return': returns_a.get('rnorm', 0),
        'sharpe_ratio': sharpe_a.get('sharperatio', 0) or 0,
        'max_drawdown': drawdown_a['max']['drawdown'] if drawdown_a.get('max') else 0,
        'trades': trades_a,
        'trade_log': strat.trade_log,
        portfolio_key: getattr(strat, portfolio_key, []),
        'elapsed_time': elapsed_time,
    }
    return result
