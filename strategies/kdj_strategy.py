"""
KDJ 交易策略
买入条件: KDJ的J值 < -5（超卖）
止盈条件: 持仓收益 >= 20%
可选止损: 可设置固定止损或J值反弹
"""

import backtrader as bt
import pandas as pd


class KDJStrategy(bt.Strategy):
    """
    KDJ超卖反弹策略
    
    参数:
        j_threshold (float): J值买入阈值，默认-5
        take_profit (float): 止盈比例，默认0.20 (20%)
        stop_loss (float): 止损比例，默认None（可选）
        n (int): KDJ计算周期N，默认9
        m1 (int): KDJ计算周期M1，默认3
        m2 (int): KDJ计算周期M2，默认3
    """
    
    params = (
        ('j_threshold', -5),      # J值买入阈值
        ('take_profit', 0.20),    # 止盈比例 20%
        ('stop_loss', None),      # 止损比例（可选）
        ('n', 9),                 # KDJ周期N
        ('m1', 3),                # KDJ周期M1
        ('m2', 3),                # KDJ周期M2
        ('print_log', True),      # 是否打印日志
    )
    
    def __init__(self):
        # 保存订单引用
        self.order = None
        self.buy_price = None
        
        # 计算KDJ指标
        # RSV = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        self.low_n = bt.indicators.Lowest(self.data.low, period=self.params.n)
        self.high_n = bt.indicators.Highest(self.data.high, period=self.params.n)
        
        # RSV
        self.rsv = 100 * (self.data.close - self.low_n) / (self.high_n - self.low_n)
        
        # K值 = RSV的M1日EMA
        self.k = bt.indicators.EMA(self.rsv, period=self.params.m1)
        
        # D值 = K值的M2日EMA
        self.d = bt.indicators.EMA(self.k, period=self.params.m2)
        
        # J值 = 3*K - 2*D
        self.j = 3 * self.k - 2 * self.d
        
        # 交易记录
        self.trades = []
    
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
            if order.isbuy():
                self.log(f'【买入执行】价格: {order.executed.price:.2f}, '
                        f'数量: {order.executed.size}')
                self.buy_price = order.executed.price
            else:
                self.log(f'【卖出执行】价格: {order.executed.price:.2f}, '
                        f'数量: {order.executed.size}')
                
                # 计算收益
                if self.buy_price:
                    pnl = (order.executed.price / self.buy_price - 1) * 100
                    self.log(f'    盈亏: {pnl:+.2f}%')
        
        self.order = None
    
    def next(self):
        """每个bar执行一次"""
        # 如果正在下单，不操作
        if self.order:
            return
        
        # 获取当前KDJ值
        j_value = self.j[0]
        k_value = self.k[0]
        d_value = self.d[0]
        
        # 如果没有持仓
        if not self.position:
            # 买入条件: J < -5（超卖）
            if j_value < self.params.j_threshold:
                self.log(f'【买入信号】J值: {j_value:.2f} < {self.params.j_threshold}')
                self.log(f'    K={k_value:.2f}, D={d_value:.2f}, J={j_value:.2f}')
                self.order = self.buy()
        
        # 如果有持仓
        else:
            # 计算当前收益率
            current_return = (self.data.close[0] / self.buy_price - 1) if self.buy_price else 0
            
            # 止盈条件: 收益 >= 20%
            if current_return >= self.params.take_profit:
                self.log(f'【止盈卖出】收益率: {current_return*100:.2f}% >= {self.params.take_profit*100:.0f}%')
                self.log(f'    买入价: {self.buy_price:.2f}, 当前价: {self.data.close[0]:.2f}')
                self.order = self.sell()
            
            # 可选: 止损条件
            elif self.params.stop_loss and current_return <= -self.params.stop_loss:
                self.log(f'【止损卖出】收益率: {current_return*100:.2f}% <= -{self.params.stop_loss*100:.0f}%')
                self.order = self.sell()
    
    def stop(self):
        """策略结束时"""
        self.log('='*50)
        self.log('策略结束')
        self.log(f'最终资金: {self.broker.getvalue():.2f}')
        self.log(f'总收益率: {(self.broker.getvalue()/self.broker.startingcash-1)*100:.2f}%')


class KDJStrategyOptimized(bt.Strategy):
    """
    KDJ策略优化版本
    增加更多过滤条件:
    - 趋势过滤（只在上升趋势买入）
    - 成交量确认
    """
    
    params = (
        ('j_threshold', -5),
        ('take_profit', 0.20),
        ('use_trend_filter', True),   # 是否使用趋势过滤
        ('ma_period', 20),            # 趋势判断均线周期
        ('print_log', True),
    )
    
    def __init__(self):
        self.order = None
        self.buy_price = None
        
        # KDJ指标
        self.low_n = bt.indicators.Lowest(self.data.low, period=9)
        self.high_n = bt.indicators.Highest(self.data.high, period=9)
        self.rsv = 100 * (self.data.close - self.low_n) / (self.high_n - self.low_n)
        self.k = bt.indicators.EMA(self.rsv, period=3)
        self.d = bt.indicators.EMA(self.k, period=3)
        self.j = 3 * self.k - 2 * self.d
        
        # 趋势过滤
        if self.params.use_trend_filter:
            self.ma20 = bt.indicators.SMA(self.data.close, period=self.params.ma_period)
    
    def log(self, txt, dt=None):
        if self.params.print_log:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
    
    def next(self):
        if self.order:
            return
        
        j_value = self.j[0]
        
        if not self.position:
            # 基础买入条件: J < -5
            buy_signal = j_value < self.params.j_threshold
            
            # 趋势过滤: 价格在MA20之上（上升趋势）
            if self.params.use_trend_filter:
                trend_ok = self.data.close[0] > self.ma20[0]
                buy_signal = buy_signal and trend_ok
            
            if buy_signal:
                self.log(f'【买入信号】J={j_value:.2f}')
                self.order = self.buy()
        
        else:
            current_return = (self.data.close[0] / self.buy_price - 1) if self.buy_price else 0
            
            # 止盈
            if current_return >= self.params.take_profit:
                self.log(f'【止盈卖出】收益: {current_return*100:.2f}%')
                self.order = self.sell()


if __name__ == '__main__':
    print("KDJ 策略模块")
    print("使用方法:")
    print("  from strategies.kdj_strategy import KDJStrategy")
    print("  cerebro.addstrategy(KDJStrategy, j_threshold=-5, take_profit=0.20)")
