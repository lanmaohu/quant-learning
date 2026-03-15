"""
TopK 策略 - 基于预测分数的选股策略

策略逻辑：
1. 每天选择预测分数最高的 TopK 只股票
2. 卖出不在 TopK 列表中的持仓
3. 买入新进入 TopK 列表的股票（等权重或按分数加权）
4. 支持再平衡频率设置

优化版本：Signals 和 Data 在一个 Loop 中处理
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class TopKConfig:
    """TopK 策略配置"""
    top_k: int = 10                   # 每天选择的股票数量
    rebalance_freq: int = 1           # 再平衡频率 (1=每日, 5=每周)
    max_positions: int = 10           # 最大持仓数量（与top_k通常一致）
    position_pct: float = 0.95        # 资金使用比例（留5%现金）
    equal_weight: bool = True         # True=等权重, False=按预测分数加权
    min_score_threshold: float = 0.0  # 最小分数阈值（低于此值不买入）
    sell_at_exit: bool = True         # 当股票掉出TopK时是否卖出


class TopKStrategy(bt.Strategy):
    """
    TopK 选股策略
    
    每天根据预测分数选择排名前K的股票，保持持仓与TopK列表一致
    """
    
    params = (
        ('config', None),           # TopKConfig 对象
        ('signals', None),          # 信号字典 {date: [(code, score), ...]}
        ('print_log', True),
    )
    
    def __init__(self):
        self.config = self.params.config or TopKConfig()
        self.signals = self.params.signals or {}
        self.counter = 0
        self.trade_log = []
        self.daily_portfolio = []  # 记录每日组合状态
        
    def log(self, txt, dt=None):
        """日志函数"""
        if self.params.print_log:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            action = '买入' if order.isbuy() else '卖出'
            self.log(f'【{action}】{order.data._name}: '
                    f'{abs(order.executed.size)}股 @ {order.executed.price:.2f} '
                    f'金额: {order.executed.size * order.executed.price:,.2f}')
            
            # 记录交易（包含买入/卖出标记用于可视化）
            self.trade_log.append({
                'date': self.datas[0].datetime.date(0),
                'code': order.data._name,
                'action': action,
                'price': order.executed.price,
                'size': abs(order.executed.size),
                'value': order.executed.size * order.executed.price,
                'type': 'buy' if order.isbuy() else 'sell'
            })
        
        self.order = None
    
    def _get_current_signals(self) -> List[Tuple[str, float]]:
        """获取当前日期的信号列表"""
        current_date = pd.Timestamp(self.datas[0].datetime.date(0))
        signals = self.signals.get(current_date, [])
        
        # 过滤低于阈值的信号
        if self.config.min_score_threshold > 0:
            signals = [(code, score) for code, score in signals 
                      if score >= self.config.min_score_threshold]
        
        return signals
    
    def _get_topk_codes(self, signals: List[Tuple[str, float]]) -> set:
        """获取TopK股票代码集合"""
        # 按分数降序排序
        sorted_signals = sorted(signals, key=lambda x: x[1], reverse=True)
        # 取前K个
        topk = sorted_signals[:self.config.top_k]
        return {code for code, _ in topk}
    
    def _get_data_by_code(self, code: str) -> Optional[bt.DataBase]:
        """根据股票代码获取data对象"""
        for d in self.datas:
            if d._name == str(code):
                return d
        return None
    
    def _get_current_positions(self) -> Dict[str, int]:
        """获取当前持仓 {code: size}"""
        positions = {}
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size > 0:
                positions[d._name] = pos.size
        return positions
    
    def next(self):
        """每日执行"""
        self.counter += 1
        
        # 记录每日组合价值
        current_date = self.datas[0].datetime.date(0)
        portfolio_value = self.broker.getvalue()
        cash = self.broker.getcash()
        
        # 按频率再平衡
        if self.counter % self.config.rebalance_freq != 0:
            # 只记录组合价值，不调仓
            self.daily_portfolio.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'n_positions': len(self._get_current_positions()),
                'rebalanced': False
            })
            return
        
        # 获取当日信号
        signals = self._get_current_signals()
        if not signals:
            self.log(f'今日无信号，跳过')
            self.daily_portfolio.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'n_positions': len(self._get_current_positions()),
                'rebalanced': False
            })
            return
        
        # 获取TopK股票
        topk_codes = self._get_topk_codes(signals)
        
        # 获取当前持仓
        positions = self._get_current_positions()
        current_codes = set(positions.keys())
        
        self.log('=' * 70)
        self.log(f'再平衡 #{self.counter} | 信号数: {len(signals)} | TopK: {len(topk_codes)}')
        self.log(f'当前持仓: {len(current_codes)}只 {list(current_codes)[:5]}{"..." if len(current_codes) > 5 else ""}')
        
        # ===== 卖出逻辑：卖出不在TopK中的持仓 =====
        sell_codes = current_codes - topk_codes
        for code in sell_codes:
            if self.config.sell_at_exit:
                data = self._get_data_by_code(code)
                if data:
                    self.close(data=data)
                    self.log(f'  📤 卖出 {code}（掉出Top{self.config.top_k}）')
        
        # ===== 计算目标仓位 =====
        portfolio_value = self.broker.getvalue()
        cash = self.broker.getcash()
        
        # 计算每只股票的 target_value
        n_targets = min(len(topk_codes), self.config.max_positions)
        if n_targets == 0:
            return
        
        investable_value = portfolio_value * self.config.position_pct
        
        if self.config.equal_weight:
            # 等权重
            target_values = {code: investable_value / n_targets for code in topk_codes}
        else:
            # 按分数加权（需要signals中的分数）
            score_dict = {code: score for code, score in signals if code in topk_codes}
            total_score = sum(score_dict.values())
            if total_score > 0:
                target_values = {code: investable_value * (score / total_score) 
                               for code, score in score_dict.items()}
            else:
                target_values = {code: investable_value / n_targets for code in topk_codes}
        
        # ===== 买入逻辑：买入新进入TopK的股票 =====
        buy_codes = topk_codes - current_codes
        
        for code in buy_codes:
            if code not in target_values:
                continue
                
            data = self._get_data_by_code(code)
            if not data:
                continue
            
            current_price = data.close[0]
            if current_price <= 0:
                continue
            
            target_value = target_values[code]
            size = int(target_value / current_price)
            
            if size > 0:
                cash_needed = size * current_price * 1.001  # 考虑手续费
                if cash_needed <= self.broker.getcash():
                    self.buy(data=data, size=size)
                    self.log(f'  📥 买入 {code}: {size}股 @ {current_price:.2f} '
                            f'目标仓位: {target_value/portfolio_value*100:.1f}%')
        
        # 记录组合状态
        self.daily_portfolio.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'positions': list(self._get_current_positions().keys()),
            'topk': list(topk_codes),
            'n_positions': len(self._get_current_positions()),
            'rebalanced': True
        })
    
    def stop(self):
        """策略结束"""
        self.log('=' * 70)
        self.log('策略结束')
        self.log(f'最终资金: {self.broker.getvalue():,.2f}')
        self.log(f'总收益率: {(self.broker.getvalue()/self.broker.startingcash-1)*100:.2f}%')


def create_trade_chart(result: Dict, title: str = "TopK Strategy", save_path: str = None):
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
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 获取数据
    portfolio_df = pd.DataFrame(result.get('daily_portfolio', []))
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
    
    # 标记再平衡日
    rebalance_df = portfolio_df[portfolio_df.get('rebalanced', False)]
    if len(rebalance_df) > 0:
        ax1.scatter(rebalance_df.index, portfolio_df.loc[rebalance_df.index, 'cum_return'],
                   marker='|', color='purple', s=50, alpha=0.5, label='Rebalance')
    
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
    
    # 绘制现金比例
    ax3_twin = ax3.twinx()
    ax3.plot(portfolio_df.index, portfolio_df.get('n_positions', 0), 
             color='blue', linewidth=1.5, label='Positions')
    ax3_twin.fill_between(portfolio_df.index, 0, portfolio_df['cash_ratio'], 
                          alpha=0.3, color='orange', label='Cash Ratio')
    
    ax3.set_ylabel('Number of Positions', fontsize=11, color='blue')
    ax3_twin.set_ylabel('Cash Ratio (%)', fontsize=11, color='orange')
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Positions & Cash', fontsize=12)
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


def run_topk_backtest(
    test_df: pd.DataFrame,
    pred_col: str = 'pred_return',
    code_col: str = 'code',
    date_col: str = 'date',
    price_col: str = 'close',
    top_k: int = 10,
    rebalance_freq: int = 1,
    equal_weight: bool = True,
    position_pct: float = 0.95,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    print_log: bool = True
) -> Dict:
    """运行 TopK 策略回测（标准版本）"""
    print("=" * 70)
    print("🚀 TopK 策略回测")
    print("=" * 70)
    print(f"\n策略参数:")
    print(f"   TopK: 每天选{top_k}只")
    print(f"   再平衡频率: 每{rebalance_freq}天")
    print(f"   权重方式: {'等权重' if equal_weight else '按分数加权'}")
    print(f"   资金使用: {position_pct*100:.0f}%")
    
    # 准备信号
    print("\n📊 准备信号数据...")
    signals = {}
    for date, group in test_df.groupby(date_col):
        daily = []
        for _, row in group.iterrows():
            daily.append((str(row[code_col]), row[pred_col]))
        signals[date] = daily
    
    config = TopKConfig(
        top_k=top_k,
        rebalance_freq=rebalance_freq,
        max_positions=top_k,
        position_pct=position_pct,
        equal_weight=equal_weight
    )
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TopKStrategy, config=config, signals=signals, print_log=print_log)
    
    # 添加数据
    codes = test_df[code_col].unique()
    n_added = 0
    for code in codes:
        stock_df = test_df[test_df[code_col] == code].copy()
        stock_df = stock_df.sort_values(date_col)
        stock_df.set_index(date_col, inplace=True)
        if len(stock_df) < 5:
            continue
        data = bt.feeds.PandasData(dataname=stock_df, datetime=None,
                                   open=price_col, high=price_col, low=price_col, close=price_col,
                                   openinterest=-1)
        cerebro.adddata(data, name=str(code))
        n_added += 1
    
    print(f"\n📈 加载了 {n_added} 只股票")
    
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print(f"\n💰 开始回测...")
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    returns_analyzer = strat.analyzers.returns.get_analysis()
    sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
    drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
    trades_analyzer = strat.analyzers.trades.get_analysis()
    
    result = {
        'strategy': 'TopK',
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': (final_value / initial_cash - 1),
        'annual_return': returns_analyzer.get('rnorm', 0),
        'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0) or 0,
        'max_drawdown': drawdown_analyzer['max']['drawdown'] if drawdown_analyzer['max'] else 0,
        'trades': trades_analyzer,
        'trade_log': strat.trade_log,
        'daily_portfolio': strat.daily_portfolio
    }
    
    print(f"\n总收益率: {result['total_return']*100:.2f}%")
    print(f"夏普比率: {result['sharpe_ratio']:.3f}")
    
    return result


def run_topk_backtest_optimized(
    test_df: pd.DataFrame,
    pred_col: str = 'pred_return',
    code_col: str = 'code',
    date_col: str = 'date',
    price_col: str = 'close',
    top_k: int = 10,
    rebalance_freq: int = 1,
    equal_weight: bool = True,
    position_pct: float = 0.95,
    min_data_days: int = 5,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    print_log: bool = True
) -> Dict:
    """运行 TopK 策略回测（优化版本 - Single Loop）"""
    start_time = time.time()
    
    print("=" * 70)
    print("🚀 TopK 策略回测（优化版 - Single Loop）")
    print("=" * 70)
    
    signals = {}
    cerebro = bt.Cerebro()
    
    all_codes = test_df[code_col].unique()
    print(f"\n📊 处理 {len(all_codes)} 只股票...")
    
    n_added = 0
    n_skipped = 0
    total_signals = 0
    
    for i, code in enumerate(all_codes):
        if print_log and (i + 1) % 50 == 0:
            print(f"   进度: {i+1}/{len(all_codes)}")
        
        stock_df = test_df[test_df[code_col] == code].copy()
        
        if len(stock_df) < min_data_days:
            n_skipped += 1
            continue
        
        stock_df = stock_df.sort_values(date_col)
        
        # 构建 Signals
        for _, row in stock_df.iterrows():
            date = row[date_col]
            score = row[pred_col]
            if date not in signals:
                signals[date] = []
            signals[date].append((str(code), score))
            total_signals += 1
        
        # 添加 Data
        stock_df.set_index(date_col, inplace=True)
        data = bt.feeds.PandasData(dataname=stock_df, datetime=None,
                                   open=price_col, high=price_col, low=price_col, close=price_col,
                                   openinterest=-1)
        cerebro.adddata(data, name=str(code))
        n_added += 1
    
    config = TopKConfig(top_k=top_k, rebalance_freq=rebalance_freq,
                       max_positions=top_k, position_pct=position_pct, equal_weight=equal_weight)
    
    cerebro.addstrategy(TopKStrategy, config=config, signals=signals, print_log=print_log)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    returns_analyzer = strat.analyzers.returns.get_analysis()
    sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
    drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
    trades_analyzer = strat.analyzers.trades.get_analysis()
    
    total_time = time.time() - start_time
    
    result = {
        'strategy': 'TopK_Optimized',
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': (final_value / initial_cash - 1),
        'annual_return': returns_analyzer.get('rnorm', 0),
        'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0) or 0,
        'max_drawdown': drawdown_analyzer['max']['drawdown'] if drawdown_analyzer['max'] else 0,
        'trades': trades_analyzer,
        'trade_log': strat.trade_log,
        'daily_portfolio': strat.daily_portfolio,
        'n_stocks': n_added,
        'n_signals': total_signals,
        'elapsed_time': total_time
    }
    
    print(f"\n总耗时: {total_time:.2f}s")
    print(f"总收益率: {result['total_return']*100:.2f}%")
    
    return result


def run_topk_backtest_ultra_optimized(
    test_df: pd.DataFrame,
    pred_col: str = 'pred_return',
    code_col: str = 'code',
    date_col: str = 'date',
    price_col: str = 'close',
    top_k: int = 10,
    rebalance_freq: int = 1,
    equal_weight: bool = True,
    position_pct: float = 0.95,
    min_data_days: int = 5,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    print_log: bool = True
) -> Dict:
    """运行 TopK 策略回测（超优化版本 - Vectorized GroupBy）"""
    start_time = time.time()
    
    print("=" * 70)
    print("🚀 TopK 策略回测（超优化版 - Vectorized GroupBy）")
    print("=" * 70)
    
    signals = {}
    cerebro = bt.Cerebro()
    
    test_df = test_df.sort_values([code_col, date_col]).copy()
    
    n_added = 0
    n_skipped = 0
    
    print("\n📊 使用 GroupBy 向量化处理...")
    
    for code, group in test_df.groupby(code_col, sort=False):
        if len(group) < min_data_days:
            n_skipped += 1
            continue
        
        dates = group[date_col].values
        scores = group[pred_col].values
        
        for date, score in zip(dates, scores):
            if date not in signals:
                signals[date] = []
            signals[date].append((str(code), float(score)))
        
        stock_df = group.set_index(date_col)
        data = bt.feeds.PandasData(dataname=stock_df, datetime=None,
                                   open=price_col, high=price_col, low=price_col, close=price_col,
                                   openinterest=-1)
        cerebro.adddata(data, name=str(code))
        n_added += 1
    
    config = TopKConfig(top_k=top_k, rebalance_freq=rebalance_freq,
                       max_positions=top_k, position_pct=position_pct, equal_weight=equal_weight)
    
    cerebro.addstrategy(TopKStrategy, config=config, signals=signals, print_log=print_log)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    returns_analyzer = strat.analyzers.returns.get_analysis()
    sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
    drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
    trades_analyzer = strat.analyzers.trades.get_analysis()
    
    total_time = time.time() - start_time
    
    result = {
        'strategy': 'TopK_Ultra',
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': (final_value / initial_cash - 1),
        'annual_return': returns_analyzer.get('rnorm', 0),
        'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0) or 0,
        'max_drawdown': drawdown_analyzer['max']['drawdown'] if drawdown_analyzer['max'] else 0,
        'trades': trades_analyzer,
        'trade_log': strat.trade_log,
        'daily_portfolio': strat.daily_portfolio,
        'elapsed_time': total_time
    }
    
    print(f"\n总耗时: {total_time:.2f}s")
    print(f"总收益率: {result['total_return']*100:.2f}%")
    
    return result


if __name__ == '__main__':
    print("TopK 策略模块")
    print("\n可用函数:")
    print("  1. run_topk_backtest() - 标准版本")
    print("  2. run_topk_backtest_optimized() - 优化版本（single loop）")
    print("  3. run_topk_backtest_ultra_optimized() - 超优化版本（vectorized groupby）")
    print("  4. create_trade_chart() - 绘制交易图表（收益曲线+买卖点）")
    print("\n推荐使用 run_topk_backtest_ultra_optimized() 获得最佳性能")
    print("使用 create_trade_chart(result) 可视化回测结果")
