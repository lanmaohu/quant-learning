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
from typing import Dict, List, Optional
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
            
            self.trade_log.append({
                'date': self.datas[0].datetime.date(0),
                'code': order.data._name,
                'action': action,
                'price': order.executed.price,
                'size': abs(order.executed.size),
                'value': abs(order.executed.size) * order.executed.price
            })
        
        self.order = None
    
    def next(self):
        """每日执行"""
        self.counter += 1
        self.log('=' * 60)
        self.log(f'  每日之星:{self.counter}')
        
        # 按频率调仓
        if self.counter % self.config.rebalance_freq != 0:
            self.log(f'  没到调仓频率')
            return
        
        current_date = pd.Timestamp(self.datas[0].datetime.date(0))
        today_signals = self.signals.get(current_date, [])
        
        if not today_signals:
            self.log(f'  今天没有signals')
            return

        self.log(f'  today signals: {len(today_signals)}')
        
        # 获取当前持仓
        positions = {}
        buy_prices = {}
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                positions[data._name] = pos.size
                buy_prices[data._name] = pos.price

        self.log(f'  buy_prices: {len(buy_prices)}')
        
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


def run_threshold_backtest(
    test_df: pd.DataFrame,
    pred_col: str = 'pred_return',
    code_col: str = 'code',
    date_col: str = 'date',
    price_col: str = 'close',
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.01,
    max_positions: int = 10,
    rebalance_freq: int = 1,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    print_log: bool = True
) -> Dict:
    """
    运行 Threshold 策略回测（标准版本 - 两次 groupby）
    """
    print("=" * 70)
    print("🚀 Threshold 策略回测（标准版）")
    print("=" * 70)
    print(f"\n策略参数:")
    print(f"   买入阈值: {buy_threshold*100:.2f}%")
    print(f"   卖出阈值: {sell_threshold*100:.2f}%")
    print(f"   最大持仓: {max_positions}只")
    print(f"   调仓频率: 每{rebalance_freq}天")

    # 准备信号 - 第一次 groupby
    signals = {}
    for date, group in test_df.groupby(date_col):
        daily = []
        for _, row in group.iterrows():
            daily.append((str(row[code_col]), row[pred_col]))
        signals[date] = daily

    config = ThresholdConfig(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_positions=max_positions,
        rebalance_freq=rebalance_freq
    )

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        ThresholdStrategy,
        config=config,
        signals=signals,
        print_log=print_log
    )
    
    # 添加数据 - 第二次遍历
    codes = test_df[code_col].unique()
    for code in codes:
        stock_df = test_df[test_df[code_col] == code].copy()
        stock_df = stock_df.sort_values(date_col)
        stock_df.set_index(date_col, inplace=True)
        
        if len(stock_df) < 5:
            continue

        data = bt.feeds.PandasData(
            dataname=stock_df,
            datetime=None,
            open=price_col, high=price_col, low=price_col, close=price_col,
            openinterest=-1
        )
        cerebro.adddata(data, name=str(code))

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print(f"\n💰 开始回测...")
    print("-" * 70)
    
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    returns_analyzer = strat.analyzers.returns.get_analysis()
    sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
    drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
    trades_analyzer = strat.analyzers.trades.get_analysis()
    
    result = {
        'strategy': 'Threshold',
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': (final_value / initial_cash - 1),
        'annual_return': returns_analyzer.get('rnorm', 0),
        'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0) or 0,
        'max_drawdown': drawdown_analyzer['max']['drawdown'] if drawdown_analyzer['max'] else 0,
        'trades': trades_analyzer,
        'trade_log': strat.trade_log
    }
    
    print("-" * 70)
    print(f"💰 最终资金: {final_value:,.2f}")
    print("\n" + "=" * 70)
    print("📊 回测结果")
    print("=" * 70)
    print(f"总收益率: {result['total_return']*100:.2f}%")
    print(f"年化收益率: {result['annual_return']*100:.2f}%")
    print(f"夏普比率: {result['sharpe_ratio']:.3f}")
    print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
    
    return result


def run_threshold_backtest_optimized(
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
    运行 Threshold 策略回测（优化版本 - Signals 和 Data 在一个 Loop 中处理）
    
    优化点：
    1. 只遍历一次股票代码，同时构建 signals 和添加 cerebro data
    2. 提前过滤数据不足的股票
    3. 减少内存拷贝
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        测试数据
    min_data_days : int
        最小数据天数要求
    其他参数同 run_threshold_backtest
    """
    start_time = time.time()
    
    print("=" * 70)
    print("🚀 Threshold 策略回测（优化版 - Single Loop）")
    print("=" * 70)
    print(f"\n策略参数:")
    print(f"   买入阈值: {buy_threshold*100:.2f}%")
    print(f"   卖出阈值: {sell_threshold*100:.2f}%")
    print(f"   最大持仓: {max_positions}只")
    print(f"   调仓频率: 每{rebalance_freq}天")
    print(f"   最小数据天数: {min_data_days}")
    
    # 初始化
    signals = {}
    cerebro = bt.Cerebro()
    
    # 获取所有唯一股票代码
    all_codes = test_df[code_col].unique()
    print(f"\n📊 开始处理 {len(all_codes)} 只股票...")
    
    n_added = 0
    n_skipped = 0
    total_signals = 0
    
    for i, code in enumerate(all_codes):
        if print_log and (i + 1) % 50 == 0:
            print(f"   处理进度: {i+1}/{len(all_codes)} ({(i+1)/len(all_codes)*100:.1f}%)")
        
        # 获取单只股票数据
        stock_df = test_df[test_df[code_col] == code].copy()
        
        # 检查数据量
        if len(stock_df) < min_data_days:
            n_skipped += 1
            continue
        
        # 排序
        stock_df = stock_df.sort_values(date_col)
        
        # ========== 1. 构建 Signals ==========
        for _, row in stock_df.iterrows():
            date = row[date_col]
            score = row[pred_col]
            
            if date not in signals:
                signals[date] = []
            signals[date].append((str(code), score))
            total_signals += 1
        
        # ========== 2. 添加 Data 到 Cerebro ==========
        stock_df.set_index(date_col, inplace=True)
        
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
    print(f"\n✅ 数据处理完成 ({elapsed:.2f}s):")
    print(f"   成功添加: {n_added} 只股票")
    print(f"   跳过(数据不足): {n_skipped} 只")
    print(f"   总信号数: {total_signals:,}")
    print(f"   交易日期: {len(signals)} 天")
    
    if n_added == 0:
        raise ValueError("没有有效的股票数据，请检查输入数据")
    
    # 创建配置
    config = ThresholdConfig(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_positions=max_positions,
        rebalance_freq=rebalance_freq
    )
    
    # 添加策略
    cerebro.addstrategy(
        ThresholdStrategy,
        config=config,
        signals=signals,
        print_log=print_log
    )
    
    # 设置经纪商
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 运行回测
    print(f"\n💰 开始回测...")
    print("-" * 70)
    
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    # 获取分析结果
    returns_analyzer = strat.analyzers.returns.get_analysis()
    sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
    drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
    trades_analyzer = strat.analyzers.trades.get_analysis()
    
    total_time = time.time() - start_time
    
    result = {
        'strategy': 'Threshold_Optimized',
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': (final_value / initial_cash - 1),
        'annual_return': returns_analyzer.get('rnorm', 0),
        'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0) or 0,
        'max_drawdown': drawdown_analyzer['max']['drawdown'] if drawdown_analyzer['max'] else 0,
        'trades': trades_analyzer,
        'trade_log': strat.trade_log,
        'n_stocks': n_added,
        'n_signals': total_signals,
        'elapsed_time': total_time
    }
    
    print("-" * 70)
    print(f"💰 最终资金: {final_value:,.2f}")
    print(f"\n📊 总耗时: {total_time:.2f}s")
    print("=" * 70)
    print("📊 回测结果")
    print("=" * 70)
    print(f"总收益率: {result['total_return']*100:.2f}%")
    print(f"年化收益率: {result['annual_return']*100:.2f}%")
    print(f"夏普比率: {result['sharpe_ratio']:.3f}")
    print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
    
    return result


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
    
    使用 Pandas GroupBy 进行向量化处理，避免 Python 层面的循环
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
    print("\n推荐使用 run_threshold_backtest_ultra() 获得最佳性能")
