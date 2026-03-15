"""
Threshold 阈值策略 - 预置版本

策略逻辑：
1. 预测收益率 > buy_threshold → 买入
2. 预测收益率 < sell_threshold → 卖出
3. 最多持有 max_positions 只股票
4. 每只股票的仓位不超过 max_position_size

优化版本：Signals 和 Data 在一个 Loop 中处理
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class ThresholdConfig:
    """Threshold 策略配置"""
    buy_threshold: float = 0.02      # 买入阈值 (2%)
    sell_threshold: float = -0.01    # 卖出阈值 (-1%)
    max_positions: int = 10          # 最大持仓数
    max_position_size: float = 0.2   # 单只股票最大仓位 (20%)
    rebalance_freq: int = 1          # 调仓频率 (1=每日, 5=每周)
    use_stop_loss: bool = False      # 是否使用止损
    stop_loss_pct: float = 0.05      # 止损比例 (5%)


class ThresholdStrategy(bt.Strategy):
    """
    Threshold 阈值交易策略
    """
    
    params = (
        ('config', None),           # ThresholdConfig 对象
        ('signals', None),          # 信号字典 {date: [(code, pred), ...]}
        ('print_log', True),
    )
    
    def __init__(self):
        self.config = self.params.config or ThresholdConfig()
        self.signals = self.params.signals or {}
        self.counter = 0
        self.trade_log = []
        self.daily_returns = []  # 每日收益记录
        self.portfolio_values = []  # 每日组合价值
        
    def log(self, txt, dt=None):
        if self.params.print_log:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
            
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            action = '买入' if order.isbuy() else '卖出'
            self.log(f'【{action}】{order.data._name}: '
                    f'{abs(order.executed.size)}股 @ {order.executed.price:.2f}')
            
            # 记录交易（包含买入/卖出标记用于可视化）
            self.trade_log.append({
                'date': self.datas[0].datetime.date(0),
                'code': order.data._name,
                'action': action,
                'price': order.executed.price,
                'size': abs(order.executed.size),
                'value': abs(order.executed.size) * order.executed.price,
                'type': 'buy' if order.isbuy() else 'sell'
            })
        
        self.order = None
    
    def next(self):
        """每日执行"""
        self.counter += 1
        
        # 记录每日组合价值
        current_date = self.datas[0].datetime.date(0)
        portfolio_value = self.broker.getvalue()
        cash = self.broker.getcash()
        
        self.portfolio_values.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': cash
        })
        
        # self.log('=' * 60)
        # self.log(f'  每日执行 #{self.counter}')
        
        # 按频率调仓
        if self.counter % self.config.rebalance_freq != 0:
            # self.log(f'  没到调仓频率')
            return
        
        current_date = pd.Timestamp(self.datas[0].datetime.date(0))
        today_signals = self.signals.get(current_date, [])
        
        if not today_signals:
            # self.log(f'  今天没有signals')
            return

        # self.log(f'  today signals: {len(today_signals)}')
        
        # 获取当前持仓
        positions = {}
        buy_prices = {}
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                positions[data._name] = pos.size
                buy_prices[data._name] = pos.price

        self.log(f'  当前持仓: {len(buy_prices)} 只')
        
        # ===== 卖出逻辑 =====
        for code, pred in today_signals:
            # 卖出条件1: 预测收益 < 卖出阈值
            sell_signal = pred < self.config.sell_threshold
            
            # 卖出条件2: 止损（如果启用)
            if self.config.use_stop_loss and code in positions:
                current_price = None
                for data in self.datas:
                    if data._name == code:
                        current_price = data.close[0]
                        break
                
                if current_price and code in buy_prices:
                    loss_pct = (current_price - buy_prices[code]) / buy_prices[code]
                    if loss_pct < -self.config.stop_loss_pct:
                        sell_signal = True
                        self.log(f'  止损信号 {code}: 亏损{loss_pct*100:.1f}%')
            
            if sell_signal and code in positions:
                for data in self.datas:
                    if data._name == code:
                        self.close(data=data)
                        self.log(f'  卖出信号 {code}: 预测{pred:.4f}')
                        break
        
        # ===== 买入逻辑 =====
        buy_candidates = [
            (code, pred) for code, pred in today_signals
            if pred > self.config.buy_threshold and code not in positions
        ]
        
        buy_candidates.sort(key=lambda x: x[1], reverse=True)
        slots_available = self.config.max_positions - len(positions)
        buy_list = buy_candidates[:slots_available]
        
        if not buy_list:
            return
        
        total_value = self.broker.getvalue()
        max_single_value = total_value * self.config.max_position_size
        
        for code, pred in buy_list:
            for data in self.datas:
                if data._name == code:
                    current_price = data.close[0]
                    pred_weight = min(pred / self.config.buy_threshold, 2.0)
                    target_value = min(max_single_value * pred_weight * 0.5, max_single_value)
                    size = int(target_value / current_price)
                    
                    if size > 0:
                        cash_needed = size * current_price * 1.001
                        if cash_needed <= self.broker.getcash():
                            self.buy(data=data, size=size)
                            self.log(f'  买入信号 {code}: 预测{pred:.4f}, 仓位{target_value/total_value*100:.1f}%')
                    break
    
    def stop(self):
        """策略结束"""
        self.log('=' * 60)
        self.log('策略结束')
        self.log(f'最终资金: {self.broker.getvalue():,.2f}')
        self.log(f'总收益率: {(self.broker.getvalue()/self.broker.startingcash-1)*100:.2f}%')


def create_trade_chart(result: Dict, title: str = "Threshold Strategy", save_path: str = None):
    """
    创建交易图表（收益曲线 + 买卖点标记）
    
    Parameters:
    -----------
    result : dict
        回测结果字典
    title : str
        图表标题
    save_path : str, optional
        保存路径
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 获取数据
    portfolio_df = pd.DataFrame(result.get('portfolio_values', []))
    if len(portfolio_df) == 0:
        print("⚠️ 没有组合价值数据，无法绘制图表")
        return
    
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df.set_index('date', inplace=True)
    
    # 计算累计收益
    initial_value = result['initial_cash']
    portfolio_df['cum_return'] = (portfolio_df['portfolio_value'] / initial_value - 1) * 100
    portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change() * 100
    
    # ===== 子图1: 累计收益曲线 + 买卖点 =====
    ax1 = axes[0]
    
    # 绘制收益曲线
    ax1.plot(portfolio_df.index, portfolio_df['cum_return'], 
             label='Cumulative Return (%)', color='steelblue', linewidth=2)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(portfolio_df.index, 0, portfolio_df['cum_return'], 
                      where=portfolio_df['cum_return'] >= 0, alpha=0.3, color='green')
    ax1.fill_between(portfolio_df.index, 0, portfolio_df['cum_return'], 
                      where=portfolio_df['cum_return'] < 0, alpha=0.3, color='red')
    
    # 标记买卖点
    trade_log = result.get('trade_log', [])
    buy_trades = [t for t in trade_log if t.get('type') == 'buy']
    sell_trades = [t for t in trade_log if t.get('type') == 'sell']
    
    # 买入点（向上三角形）
    for trade in buy_trades:
        date = pd.to_datetime(trade['date'])
        if date in portfolio_df.index:
            value = portfolio_df.loc[date, 'cum_return']
            ax1.scatter(date, value, marker='^', color='red', s=100, 
                       zorder=5, label='Buy' if trade == buy_trades[0] else "")
    
    # 卖出点（向下三角形）
    for trade in sell_trades:
        date = pd.to_datetime(trade['date'])
        if date in portfolio_df.index:
            value = portfolio_df.loc[date, 'cum_return']
            ax1.scatter(date, value, marker='v', color='green', s=100, 
                       zorder=5, label='Sell' if trade == sell_trades[0] else "")
    
    ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax1.set_title(f'{title} - Return Curve & Trade Points', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息文本框
    stats_text = f"""
    Total Return: {result['total_return']*100:.2f}%
    Annual Return: {result.get('annual_return', 0)*100:.2f}%
    Sharpe Ratio: {result['sharpe_ratio']:.2f}
    Max Drawdown: {result['max_drawdown']*100:.2f}%
    Trades: {len(trade_log)}
    """
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ===== 子图2: 每日收益柱状图 =====
    ax2 = axes[1]
    colors = ['green' if r >= 0 else 'red' for r in portfolio_df['daily_return']]
    ax2.bar(portfolio_df.index, portfolio_df['daily_return'], color=colors, alpha=0.6)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Daily Return (%)', fontsize=11)
    ax2.set_title('Daily Returns', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # ===== 子图3: 持仓数量 & 现金比例 =====
    ax3 = axes[2]
    portfolio_df['cash_ratio'] = portfolio_df['cash'] / portfolio_df['portfolio_value'] * 100
    ax3.plot(portfolio_df.index, portfolio_df['cash_ratio'], 
             color='orange', linewidth=1.5, label='Cash Ratio (%)')
    ax3.fill_between(portfolio_df.index, 0, portfolio_df['cash_ratio'], alpha=0.3, color='orange')
    ax3.set_ylabel('Cash Ratio (%)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Cash Position', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 图表已保存: {save_path}")
    
    plt.show()


# def run_threshold_backtest(
#     test_df: pd.DataFrame,
#     pred_col: str = 'pred_return',
#     code_col: str = 'code',
#     date_col: str = 'date',
#     price_col: str = 'close',
#     buy_threshold: float = 0.02,
#     sell_threshold: float = -0.01,
#     max_positions: int = 10,
#     rebalance_freq: int = 1,
#     initial_cash: float = 100000.0,
#     commission: float = 0.001,
#     print_log: bool = True
# ) -> Dict:
#     """
#     运行 Threshold 策略回测（标准版本 - 两次 groupby）
#     """
#     print("=" * 70)
#     print("🚀 Threshold 策略回测（标准版）")
#     print("=" * 70)
#     print(f"\n策略参数:")
#     print(f"   买入阈值: {buy_threshold*100:.2f}%")
#     print(f"   卖出阈值: {sell_threshold*100:.2f}%")
#     print(f"   最大持仓: {max_positions}只")
#     print(f"   调仓频率: 每{rebalance_freq}天")

#     # 准备信号 - 第一次 groupby
#     signals = {}
#     for date, group in test_df.groupby(date_col):
#         daily = []
#         for _, row in group.iterrows():
#             daily.append((str(row[code_col]), row[pred_col]))
#         signals[date] = daily

#     config = ThresholdConfig(
#         buy_threshold=buy_threshold,
#         sell_threshold=sell_threshold,
#         max_positions=max_positions,
#         rebalance_freq=rebalance_freq
#     )

#     cerebro = bt.Cerebro()
#     cerebro.addstrategy(
#         ThresholdStrategy,
#         config=config,
#         signals=signals,
#         print_log=print_log
#     )
    
#     # 添加数据 - 第二次遍历
#     codes = test_df[code_col].unique()
#     for code in codes:
#         stock_df = test_df[test_df[code_col] == code].copy()
#         stock_df = stock_df.sort_values(date_col)
#         stock_df.set_index(date_col, inplace=True)
        
#         if len(stock_df) < 5:
#             continue

#         data = bt.feeds.PandasData(
#             dataname=stock_df,
#             datetime=None,
#             open=price_col, high=price_col, low=price_col, close=price_col,
#             openinterest=-1
#         )
#         cerebro.adddata(data, name=str(code))

#     cerebro.broker.setcash(initial_cash)
#     cerebro.broker.setcommission(commission=commission)
    
#     cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
#     cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
#     cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
#     cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

#     print(f"\n💰 开始回测...")
#     print("-" * 70)
    
#     results = cerebro.run()
#     strat = results[0]
#     final_value = cerebro.broker.getvalue()
    
#     returns_analyzer = strat.analyzers.returns.get_analysis()
#     sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
#     drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
#     trades_analyzer = strat.analyzers.trades.get_analysis()
    
#     result = {
#         'strategy': 'Threshold',
#         'initial_cash': initial_cash,
#         'final_value': final_value,
#         'total_return': (final_value / initial_cash - 1),
#         'annual_return': returns_analyzer.get('rnorm', 0),
#         'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0) or 0,
#         'max_drawdown': drawdown_analyzer['max']['drawdown'] if drawdown_analyzer['max'] else 0,
#         'trades': trades_analyzer,
#         'trade_log': strat.trade_log,
#         'portfolio_values': strat.portfolio_values  # 每日组合价值
#     }
    
#     print("-" * 70)
#     print(f"💰 最终资金: {final_value:,.2f}")
#     print("\n" + "=" * 70)
#     print("📊 回测结果")
#     print("=" * 70)
#     print(f"总收益率: {result['total_return']*100:.2f}%")
#     print(f"年化收益率: {result['annual_return']*100:.2f}%")
#     print(f"夏普比率: {result['sharpe_ratio']:.3f}")
#     print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
    
#     return result


# def run_threshold_backtest_optimized(
#     test_df: pd.DataFrame,
#     pred_col: str = 'pred_return',
#     code_col: str = 'code',
#     date_col: str = 'date',
#     price_col: str = 'close',
#     buy_threshold: float = 0.02,
#     sell_threshold: float = -0.01,
#     max_positions: int = 10,
#     rebalance_freq: int = 1,
#     min_data_days: int = 5,
#     initial_cash: float = 100000.0,
#     commission: float = 0.001,
#     print_log: bool = True
# ) -> Dict:
#     """
#     运行 Threshold 策略回测（优化版本 - Signals 和 Data 在一个 Loop 中处理）
#     """
#     start_time = time.time()
    
#     print("=" * 70)
#     print("🚀 Threshold 策略回测（优化版 - Single Loop）")
#     print("=" * 70)
#     print(f"\n策略参数:")
#     print(f"   买入阈值: {buy_threshold*100:.2f}%")
#     print(f"   卖出阈值: {sell_threshold*100:.2f}%")
#     print(f"   最大持仓: {max_positions}只")
#     print(f"   调仓频率: 每{rebalance_freq}天")
#     print(f"   最小数据天数: {min_data_days}")
    
#     # 初始化
#     signals = {}
#     cerebro = bt.Cerebro()
    
#     # 获取所有唯一股票代码
#     all_codes = test_df[code_col].unique()
#     print(f"\n📊 开始处理 {len(all_codes)} 只股票...")
    
#     n_added = 0
#     n_skipped = 0
#     total_signals = 0
    
#     for i, code in enumerate(all_codes):
#         if print_log and (i + 1) % 50 == 0:
#             print(f"   处理进度: {i+1}/{len(all_codes)} ({(i+1)/len(all_codes)*100:.1f}%)")
        
#         # 获取单只股票数据
#         stock_df = test_df[test_df[code_col] == code].copy()
        
#         # 检查数据量
#         if len(stock_df) < min_data_days:
#             n_skipped += 1
#             continue
        
#         # 排序
#         stock_df = stock_df.sort_values(date_col)
        
#         # ========== 1. 构建 Signals ==========
#         for _, row in stock_df.iterrows():
#             date = row[date_col]
#             score = row[pred_col]
            
#             if date not in signals:
#                 signals[date] = []
#             signals[date].append((str(code), score))
#             total_signals += 1
        
#         # ========== 2. 添加 Data 到 Cerebro ==========
#         stock_df.set_index(date_col, inplace=True)
        
#         data = bt.feeds.PandasData(
#             dataname=stock_df,
#             datetime=None,
#             open=price_col, 
#             high=price_col, 
#             low=price_col, 
#             close=price_col,
#             openinterest=-1
#         )
#         cerebro.adddata(data, name=str(code))
#         n_added += 1
    
#     elapsed = time.time() - start_time
#     print(f"\n✅ 数据处理完成 ({elapsed:.2f}s):")
#     print(f"   成功添加: {n_added} 只股票")
#     print(f"   跳过(数据不足): {n_skipped} 只")
#     print(f"   总信号数: {total_signals:,}")
#     print(f"   交易日期: {len(signals)} 天")
    
#     if n_added == 0:
#         raise ValueError("没有有效的股票数据，请检查输入数据")
    
#     # 创建配置
#     config = ThresholdConfig(
#         buy_threshold=buy_threshold,
#         sell_threshold=sell_threshold,
#         max_positions=max_positions,
#         rebalance_freq=rebalance_freq
#     )
    
#     # 添加策略
#     cerebro.addstrategy(
#         ThresholdStrategy,
#         config=config,
#         signals=signals,
#         print_log=print_log
#     )
    
#     # 设置经纪商
#     cerebro.broker.setcash(initial_cash)
#     cerebro.broker.setcommission(commission=commission)
    
#     # 添加分析器
#     cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
#     cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
#     cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
#     cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
#     # 运行回测
#     print(f"\n💰 开始回测...")
#     print("-" * 70)
    
#     results = cerebro.run()
#     strat = results[0]
#     final_value = cerebro.broker.getvalue()
    
#     # 获取分析结果
#     returns_analyzer = strat.analyzers.returns.get_analysis()
#     sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
#     drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
#     trades_analyzer = strat.analyzers.trades.get_analysis()
    
#     total_time = time.time() - start_time
    
#     result = {
#         'strategy': 'Threshold_Optimized',
#         'initial_cash': initial_cash,
#         'final_value': final_value,
#         'total_return': (final_value / initial_cash - 1),
#         'annual_return': returns_analyzer.get('rnorm', 0),
#         'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0) or 0,
#         'max_drawdown': drawdown_analyzer['max']['drawdown'] if drawdown_analyzer['max'] else 0,
#         'trades': trades_analyzer,
#         'trade_log': strat.trade_log,
#         'portfolio_values': strat.portfolio_values,
#         'n_stocks': n_added,
#         'n_signals': total_signals,
#         'elapsed_time': total_time
#     }
    
#     print("-" * 70)
#     print(f"💰 最终资金: {final_value:,.2f}")
#     print(f"\n📊 总耗时: {total_time:.2f}s")
#     print("=" * 70)
#     print("📊 回测结果")
#     print("=" * 70)
#     print(f"总收益率: {result['total_return']*100:.2f}%")
#     print(f"年化收益率: {result['annual_return']*100:.2f}%")
#     print(f"夏普比率: {result['sharpe_ratio']:.3f}")
#     print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
    
#     return result


def run_threshold_backtest_ultra(
    test_df: pd.DataFrame,
    pred_col: str = 'pred_return',
    code_col: str = 'code',
    date_col: str = 'date',
    price_col: str = 'close',
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.01,
    max_positions: int = 10,
    rebalance_freq: int = 1,
    min_data_days: int = 5,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    print_log: bool = True
) -> Dict:
    """
    运行 Threshold 策略回测（超优化版本 - Vectorized GroupBy）
    """
    start_time = time.time()
    
    print("=" * 70)
    print("🚀 Threshold 策略回测（超优化版 - Vectorized GroupBy）")
    print("=" * 70)
    
    # 初始化
    signals = {}
    cerebro = bt.Cerebro()
    
    # 预处理：排序
    test_df = test_df.sort_values([code_col, date_col]).copy()
    
    # 使用 GroupBy 高效处理
    n_added = 0
    n_skipped = 0
    
    print("\n📊 使用 GroupBy 向量化处理...")
    
    for code, group in test_df.groupby(code_col, sort=False):
        if len(group) < min_data_days:
            n_skipped += 1
            continue
        
        # 1. 构建 Signals（向量化）
        dates = group[date_col].values
        scores = group[pred_col].values
        
        for date, score in zip(dates, scores):
            if date not in signals:
                signals[date] = []
            signals[date].append((str(code), float(score)))
        
        # 2. 添加 Data（需要设置索引）
        stock_df = group.set_index(date_col)
        
        data = bt.feeds.PandasData(
            dataname=stock_df,
            datetime=None,
            open=price_col, 
            high=price_col, 
            low=price_col, 
            close=price_col,
            openinterest=-1
        )
        cerebro.adddata(data, name=str(code))
        n_added += 1
    
    elapsed = time.time() - start_time
    print(f"✅ 数据处理完成 ({elapsed:.2f}s):")
    print(f"   成功添加: {n_added} 只股票")
    print(f"   跳过: {n_skipped} 只")
    print(f"   信号日期: {len(signals)} 天")
    
    # 创建配置和策略
    config = ThresholdConfig(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_positions=max_positions,
        rebalance_freq=rebalance_freq
    )
    
    cerebro.addstrategy(
        ThresholdStrategy, 
        config=config, 
        signals=signals, 
        print_log=print_log
    )
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 运行
    print(f"\n💰 开始回测...")
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    # 结果
    returns_analyzer = strat.analyzers.returns.get_analysis()
    sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
    drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
    trades_analyzer = strat.analyzers.trades.get_analysis()
    
    total_time = time.time() - start_time
    
    result = {
        'strategy': 'Threshold_Ultra',
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': (final_value / initial_cash - 1),
        'annual_return': returns_analyzer.get('rnorm', 0),
        'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0) or 0,
        'max_drawdown': drawdown_analyzer['max']['drawdown'] if drawdown_analyzer['max'] else 0,
        'trades': trades_analyzer,
        'trade_log': strat.trade_log,
        'portfolio_values': strat.portfolio_values,
        'elapsed_time': total_time
    }
    
    print(f"\n📊 总耗时: {total_time:.2f}s")
    print(f"总收益率: {result['total_return']*100:.2f}%")
    print(f"夏普比率: {result['sharpe_ratio']:.3f}")
    
    return result


if __name__ == '__main__':
    print("Threshold 策略模块")
    print("\n可用函数:")
    print("  1. run_threshold_backtest() - 标准版本（两次groupby）")
    print("  2. run_threshold_backtest_optimized() - 优化版本（single loop）")
    print("  3. run_threshold_backtest_ultra() - 超优化版本（vectorized groupby）")
    print("  4. create_trade_chart() - 绘制交易图表（收益曲线+买卖点）")
    print("\n推荐使用 run_threshold_backtest_ultra() 获得最佳性能")
    print("使用 create_trade_chart(result) 可视化回测结果")


def analyze_trade_profit_loss(result: Dict, save_path: str = None):
    """
    分析所有交易的盈亏情况
    
    Parameters:
    -----------
    result : dict
        回测结果字典（包含 trade_log）
    save_path : str, optional
        保存图表路径
    
    Returns:
    --------
    analysis_df : pd.DataFrame
        每笔交易的盈亏明细
    """
    import matplotlib.pyplot as plt
    
    trade_log = result.get('trade_log', [])
    if not trade_log:
        print("⚠️ 没有交易记录")
        return None
    
    # 将交易记录转为DataFrame
    trades_df = pd.DataFrame(trade_log)
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    
    # 匹配买入和卖出记录
    buy_trades = trades_df[trades_df['type'] == 'buy'].copy()
    sell_trades = trades_df[trades_df['type'] == 'sell'].copy()
    
    # 按股票代码匹配买卖记录
    completed_trades = []
    
    for code in trades_df['code'].unique():
        code_buys = buy_trades[buy_trades['code'] == code].sort_values('date')
        code_sells = sell_trades[sell_trades['code'] == code].sort_values('date')
        
        # 简单的 FIFO 匹配
        for idx, sell in code_sells.iterrows():
            # 找到这次卖出之前的买入记录
            matching_buys = code_buys[code_buys['date'] < sell['date']]
            if len(matching_buys) > 0:
                buy = matching_buys.iloc[-1]  # 取最近的一次买入
                
                # 计算盈亏
                buy_value = buy['size'] * buy['price']
                sell_value = sell['size'] * sell['price']
                profit = sell_value - buy_value
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
                
                # 移除已匹配的买入记录
                code_buys = code_buys.drop(buy.name)
    
    if not completed_trades:
        print("⚠️ 没有完成的交易对（需要有买入和卖出记录）")
        return None
    
    analysis_df = pd.DataFrame(completed_trades)
    
    # ===== 打印统计信息 =====
    print("\n" + "=" * 70)
    print("📊 交易盈亏分析")
    print("=" * 70)
    
    total_trades = len(analysis_df)
    winning_trades = len(analysis_df[analysis_df['profit'] > 0])
    losing_trades = len(analysis_df[analysis_df['profit'] < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    total_profit = analysis_df['profit'].sum()
    avg_profit = analysis_df['profit'].mean()
    avg_profit_pct = analysis_df['profit_pct'].mean()
    
    print(f"\n总交易次数: {total_trades}")
    print(f"盈利次数: {winning_trades} ({win_rate:.1f}%)")
    print(f"亏损次数: {losing_trades} ({100-win_rate:.1f}%)")
    print(f"\n总盈亏: {total_profit:,.2f}")
    print(f"平均盈亏: {avg_profit:,.2f}")
    print(f"平均收益率: {avg_profit_pct:.2f}%")
    
    if winning_trades > 0:
        avg_win = analysis_df[analysis_df['profit'] > 0]['profit'].mean()
        max_win = analysis_df[analysis_df['profit'] > 0]['profit'].max()
        print(f"\n平均盈利: {avg_win:,.2f}")
        print(f"最大盈利: {max_win:,.2f}")
    
    if losing_trades > 0:
        avg_loss = analysis_df[analysis_df['profit'] < 0]['profit'].mean()
        max_loss = analysis_df[analysis_df['profit'] < 0]['profit'].min()
        print(f"平均亏损: {avg_loss:,.2f}")
        print(f"最大亏损: {max_loss:,.2f}")
    
    print(f"\n平均持仓天数: {analysis_df['holding_days'].mean():.1f}天")
    
    # ===== 显示每笔交易明细 =====
    print("\n" + "-" * 70)
    print("📋 交易明细（按盈亏排序）")
    print("-" * 70)
    
    # 按盈亏降序排列
    analysis_sorted = analysis_df.sort_values('profit', ascending=False)
    
    print(f"{'排名':<4} {'代码':<10} {'买入价':<10} {'卖出价':<10} {'股数':<10} {'收益率':<10} {'盈亏':<12} {'持仓天数':<8}")
    print("-" * 90)
    
    for i, (_, row) in enumerate(analysis_sorted.iterrows(), 1):
        profit_str = f"{row['profit']:>+.2f}"
        print(f"{i:<4} {row['code']:<10} "
              f"{row['buy_price']:<10.2f} "
              f"{row['sell_price']:<10.2f} "
              f"{int(row['size']):<10} "
              f"{row['profit_pct']:>+8.2f}% "
              f"{profit_str:<12} {int(row['holding_days']):<8}")
    
    print("=" * 70)
    
    # ===== 可视化 =====
    if save_path is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 盈亏分布直方图
        ax1 = axes[0, 0]
        colors = ['green' if p > 0 else 'red' for p in analysis_df['profit_pct']]
        ax1.bar(range(len(analysis_df)), analysis_df['profit_pct'].sort_values(ascending=True), 
                color=['green' if p > 0 else 'red' for p in analysis_df['profit_pct'].sort_values(ascending=True)])
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
        analysis_sorted_by_date = analysis_df.sort_values('sell_date')
        analysis_sorted_by_date['cum_profit'] = analysis_sorted_by_date['profit'].cumsum()
        ax3.plot(analysis_sorted_by_date['sell_date'], analysis_sorted_by_date['cum_profit'], 
                linewidth=2, color='steelblue')
        ax3.fill_between(analysis_sorted_by_date['sell_date'], 0, 
                        analysis_sorted_by_date['cum_profit'],
                        where=analysis_sorted_by_date['cum_profit'] >= 0, alpha=0.3, color='green')
        ax3.fill_between(analysis_sorted_by_date['sell_date'], 0, 
                        analysis_sorted_by_date['cum_profit'],
                        where=analysis_sorted_by_date['cum_profit'] < 0, alpha=0.3, color='red')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative Profit')
        ax3.set_title('Cumulative Trade P&L')
        ax3.grid(True, alpha=0.3)
        
        # 4. 盈亏饼图
        ax4 = axes[1, 1]
        labels = ['Win', 'Loss']
        sizes = [winning_trades, losing_trades]
        colors = ['#66b3ff', '#ff9999']
        explode = (0.05, 0)
        ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax4.set_title(f'Win Rate ({win_rate:.1f}%)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 盈亏分析图表已保存: {save_path}")
        plt.show()
    
    return analysis_df
