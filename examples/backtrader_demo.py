"""
Backtrader 入门示例
双均线策略：当短期均线上穿长期均线时买入，下穿时卖出
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt
import pandas as pd
from datetime import datetime
from utils.data_fetcher import DataFetcher
from utils.constants import DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE


class DualMAStrategy(bt.Strategy):
    """
    双均线策略
    """
    params = (
        ('short_period', 20),   # 短期均线周期
        ('long_period', 60),    # 长期均线周期
    )
    
    def __init__(self):
        # 初始化订单状态
        self.order = None
        
        # 计算均线
        self.short_ma = bt.indicators.SMA(self.data.close, period=self.params.short_period)
        self.long_ma = bt.indicators.SMA(self.data.close, period=self.params.long_period)
        
        # 计算交叉信号
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)
        
        # 记录交易
        self.trade_log = []
    
    def next(self):
        # 如果正在下单，不操作
        if self.order:
            return
        
        # 金叉买入
        if self.crossover > 0:
            if not self.position:  # 没有持仓才买入
                self.order = self.buy()
                self.trade_log.append({
                    'date': self.data.datetime.date(0),
                    'action': 'BUY',
                    'price': self.data.close[0],
                    'size': self.getsizing()
                })
                print(f"📈 买入信号: {self.data.datetime.date(0)}, 价格: {self.data.close[0]:.2f}")
        
        # 死叉卖出
        elif self.crossover < 0:
            if self.position:  # 有持仓才卖出
                self.order = self.sell()
                self.trade_log.append({
                    'date': self.data.datetime.date(0),
                    'action': 'SELL',
                    'price': self.data.close[0],
                    'size': self.position.size
                })
                print(f"📉 卖出信号: {self.data.datetime.date(0)}, 价格: {self.data.close[0]:.2f}")
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"✅ 买入执行: 价格 {order.executed.price:.2f}, 数量 {order.executed.size}")
            else:
                print(f"✅ 卖出执行: 价格 {order.executed.price:.2f}, 数量 {order.executed.size}")
        
        self.order = None
    
    def stop(self):
        # 策略结束时打印收益
        print(f"\n📊 策略结束")
        print(f"   总收益率: {(self.broker.getvalue() / 100000 - 1) * 100:.2f}%")


def run_backtest(symbol='000001', start_date='20220101', end_date='20241231'):
    """
    运行回测
    """
    print("=" * 60)
    print("🚀 Backtrader 双均线策略回测")
    print("=" * 60)
    
    # 1. 获取数据
    print(f"\n📥 获取 {symbol} 数据...")
    fetcher = DataFetcher()
    df = fetcher.get_daily_data_ak(symbol, start_date, end_date)
    
    # 2. 准备Backtrader数据
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # 使用索引
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    
    # 3. 初始化Cerebro引擎
    cerebro = bt.Cerebro()
    
    # 添加数据
    cerebro.adddata(data, name=symbol)
    
    # 添加策略
    cerebro.addstrategy(DualMAStrategy, short_period=20, long_period=60)
    
    # 设置初始资金
    cerebro.broker.setcash(100000.0)
    
    # 设置佣金
    cerebro.broker.setcommission(commission=DEFAULT_COMMISSION_RATE)
    
    # 设置滑点
    cerebro.broker.set_slippage_perc(DEFAULT_SLIPPAGE)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 4. 运行回测
    print(f"\n💰 初始资金: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    strat = results[0]
    print(f"💰 最终资金: {cerebro.broker.getvalue():.2f}")
    
    # 5. 打印分析结果
    print("\n" + "=" * 60)
    print("📈 回测结果分析")
    print("=" * 60)
    
    # 收益率
    returns = strat.analyzers.returns.get_analysis()
    print(f"总收益率: {returns['rtot'] * 100:.2f}%")
    print(f"年化收益率: {returns['rnorm'] * 100:.2f}%")
    
    # 夏普比率
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"夏普比率: {sharpe.get('sharperatio', 0):.3f}")
    
    # 最大回撤
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"最大回撤: {drawdown['max']['drawdown']:.2f}%")
    
    # 交易统计
    trades = strat.analyzers.trades.get_analysis()
    if trades.get('total', {}).get('total', 0) > 0:
        print(f"总交易次数: {trades['total']['total']}")
        print(f"盈利次数: {trades.get('won', {}).get('total', 0)}")
        print(f"亏损次数: {trades.get('lost', {}).get('total', 0)}")
    
    # 6. 绘图（可选）
    # cerebro.plot(style='candlestick')
    
    return strat


if __name__ == '__main__':
    # 运行平安银行的回测
    run_backtest(symbol='000001', start_date='20220101', end_date='20241231')
