"""
Threshold 阈值策略 - 预置版本

策略逻辑：
1. 预测收益率 > buy_threshold → 买入
2. 预测收益率 < sell_threshold → 卖出
3. 最多持有 max_positions 只股票
4. 每只股票的仓位不超过 max_position_size
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


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
    运行 Threshold 策略回测（简化接口)
    """
    print("=" * 70)
    print("🚀 Threshold 策略回测")
    print("=" * 70)
    print(f"\n策略参数:")
    print(f"   买入阈值: {buy_threshold*100:.2f}%")
    print(f"   卖出阈值: {sell_threshold*100:.2f}%")
    print(f"   最大持仓: {max_positions}只")
    print(f"   调仓频率: 每{rebalance_freq}天")

    print(f"总行数: {len(test_df)}")
    print(f"每列缺失值数量:\n{test_df.isnull().sum()}")
    
    # 重点检查：去掉任何一行有缺失值的行后，还剩多少？
    clean_df = test_df.dropna()
    print(f"去除 NaN 后的有效行数: {len(clean_df)}")

    # 准备信号
    signals = {}
    for date, group in test_df.groupby(date_col):
        daily = []
        for _, row in group.iterrows():
            daily.append((str(row[code_col]), row[pred_col]))
        signals[date] = daily

    # 假设 signals 已经生成
    print(f"{'='*70}")
    print("📊 Signals 信号字典检查")
    print(f"{'='*70}")
    
    # 1. 基本信息
    print(f"\n📋 基本信息:")
    print(f"   总交易日数: {len(signals)}")
    print(f"   日期范围: {min(signals.keys())} ~ {max(signals.keys())}")
    
    # 2. 打印前5天的详细信号
    print(f"\n📅 前5个交易日的信号:")
    for i, (date, daily_signals) in enumerate(sorted(signals.items())[:5]):
      print(f"\n   {date} ({len(daily_signals)}只股票):")
      # 排序显示前10只
      sorted_daily = sorted(daily_signals, key=lambda x: x[1], reverse=True)[:10]
      for code, pred in sorted_daily:
          print(f"      {code}: {pred:.6f}")
    
    # 3. 统计每天的股票数量
    signal_counts = {date: len(daily) for date, daily in signals.items()}
    counts_df = pd.DataFrame(list(signal_counts.items()), columns=['date', 'stock_count'])
    
    print(f"\n📈 每日信号数量统计:")
    print(f"   最小: {counts_df['stock_count'].min()}")
    print(f"   最大: {counts_df['stock_count'].max()}")
    print(f"   平均: {counts_df['stock_count'].mean():.1f}")
    print(f"   中位数: {counts_df['stock_count'].median()}")
    
    # 4. 所有预测值的分布
    all_preds = [pred for daily in signals.values() for _, pred in daily]
    import numpy as np
    print(f"\n🔮 所有预测值分布:")
    print(f"   总数: {len(all_preds)}")
    print(f"   最小值: {min(all_preds):.6f}")
    print(f"   最大值: {max(all_preds):.6f}")
    print(f"   均值: {np.mean(all_preds):.6f}")
    print(f"   中位数: {np.median(all_preds):.6f}")
    print(f"   标准差: {np.std(all_preds):.6f}")
    print(f"   正预测比例: {sum(1 for p in all_preds if p > 0) / len(all_preds) * 100:.2f}%")
    
    # 5. 保存到CSV（方便查看）
    signals_export = []
    for date, daily in signals.items():
      for code, pred in daily:
          signals_export.append({'date': date, 'code': code, 'pred_return': pred})
    
    signals_df = pd.DataFrame(signals_export)
    output_file = './data/signals_check.csv'
    signals_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 已导出到: {output_file}")
    print(f"   总行数: {len(signals_df)}")
    
    
    config = ThresholdConfig(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_positions=max_positions,
        rebalance_freq=rebalance_freq
    )


    print(f"   signals length: {len(signals)}天")
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        ThresholdStrategy,
        config=config,
        signals=signals,
        print_log=print_log
    )
    
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


    print(f"\n{'='*60}")
    print("📊 Cerebro 数据检查")
    print(f"{'='*60}")
    
    # 1. 总共加载了多少只股票
    print(f"\n加载的股票总数: {len(cerebro.datas)}")
    
    
    # 4. 检查是否有数据为空的
    empty_data = [d for d in cerebro.datas if len(d) == 0]
    print(f"\n空数据股票数: {len(empty_data)}")
    
    # 5. 按数据条数排序
    data_lengths = [(d._name, len(d)) for d in cerebro.datas]
    data_lengths.sort(key=lambda x: x[1])
    print(f"\n数据量最少的前5只:")
    for name, length in data_lengths[:5]:
      print(f"  {name}: {length} 条")




      
    

    
    print(f"\n数据概览:")
    print(f"   股票数: {len(codes)}")
    print(f"   交易日: {test_df[date_col].nunique()}天")
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
