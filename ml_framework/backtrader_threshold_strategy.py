"""
Threshold 阈值策略 - Backtrader 实现
买入条件: 预测收益 > buy_threshold
卖出条件: 预测收益 < sell_threshold 或达到止损
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ThresholdConfig:
    """阈值策略配置"""
    buy_threshold: float = 0.02      # 买入阈值
    sell_threshold: float = -0.01    # 卖出阈值
    max_positions: int = 10          # 最大持仓数
    max_position_size: float = 0.2   # 单只股票最大仓位
    rebalance_freq: int = 1          # 调仓频率
    use_stop_loss: bool = False      # 是否使用止损
    stop_loss_pct: float = 0.05      # 止损比例


class ThresholdStrategy(bt.Strategy):
    """阈值交易策略"""
    
    params = (
        ('config', None),
        ('signals', None),
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
        
        if self.counter % self.config.rebalance_freq != 0:
            return
        
        current_date = pd.Timestamp(self.datas[0].datetime.date(0))
        today_signals = self.signals.get(current_date, [])
        
        if not today_signals:
            return
        
        # 获取当前持仓
        positions = {}
        buy_prices = {}
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                positions[data._name] = pos.size
                buy_prices[data._name] = pos.price
        
        # ===== 卖出逻辑 =====
        for code, pred in today_signals:
            sell_signal = pred < self.config.sell_threshold
            
            # 止损检查
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
                            self.log(f'  买入信号 {code}: 预测{pred:.4f}')
                    break
    
    def stop(self):
        self.log('=' * 60)
        self.log('策略结束')
        self.log(f'最终资金: {self.broker.getvalue():.2f}')


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
    max_stocks: int = 100,
    print_log: bool = True
) -> Dict:
    """运行 Threshold 策略回测"""
    
    print("=" * 70)
    print("🚀 Threshold 策略回测")
    print("=" * 70)
    print(f"\n策略参数:")
    print(f"   买入阈值: {buy_threshold*100:.2f}%")
    print(f"   卖出阈值: {sell_threshold*100:.2f}%")
    print(f"   最大持仓: {max_positions}只")
    print(f"   调仓频率: 每{rebalance_freq}天")
    
    # 准备信号
    signals = {}
    for date, group in test_df.groupby(date_col):
        daily = []
        for _, row in group.iterrows():
            daily.append((str(row[code_col]), row[pred_col]))
        signals[date] = daily
    
    # 打印信号统计
    print(f"\n📊 信号统计:")
    print(f"   总交易日: {len(signals)}天")
    all_preds = [pred for daily in signals.values() for _, pred in daily]
    print(f"   总预测数: {len(all_preds)}")
    print(f"   预测均值: {np.mean(all_preds):.6f}")
    print(f"   正预测比例: {sum(1 for p in all_preds if p > 0) / len(all_preds) * 100:.1f}%")
    
    config = ThresholdConfig(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_positions=max_positions,
        rebalance_freq=rebalance_freq
    )
    
    # 初始化 Cerebro
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        ThresholdStrategy,
        config=config,
        signals=signals,
        print_log=print_log
    )
    
    # 只加载有信号的股票
    signal_codes = set()
    for daily in signals.values():
        for code, _ in daily:
            signal_codes.add(code)
    
    codes = list(signal_codes)[:max_stocks]
    
    print(f"\n📂 加载数据:")
    print(f"   有信号的股票: {len(signal_codes)}只")
    print(f"   实际加载: {len(codes)}只")
    
    loaded_count = 0
    for code in codes:
        stock_df = test_df[test_df[code_col] == code].copy()
        
        if len(stock_df) < 5:
            continue
        
        stock_df = stock_df.sort_values(date_col)
        stock_df.set_index(date_col, inplace=True)
        
        try:
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
            loaded_count += 1
        except Exception as e:
            if print_log:
                print(f"   ⚠️ 加载 {code} 失败: {e}")
    
    print(f"   成功加载: {loaded_count}只")
    
    # Cerebro 数据检查（安全版本）
    print(f"\n{'='*60}")
    print("📊 Cerebro 数据检查")
    print(f"{'='*60}")
    print(f"\n加载的股票总数: {len(cerebro.datas)}")
    
    # 打印前5只股票的信息
    print(f"\n前5只股票:")
    for i, data in enumerate(cerebro.datas[:5]):
        data_len = len(data)
        print(f"  {i+1}. {data._name}: {data_len}条数据")
    
    # 检查空数据
    empty_data = [d for d in cerebro.datas if len(d) == 0]
    print(f"\n空数据股票数: {len(empty_data)}")
    
    # 设置资金和手续费
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 运行
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
    
    if trades_analyzer and trades_analyzer.get('total'):
        total = trades_analyzer['total']['total']
        won = trades_analyzer.get('won', {}).get('total', 0)
        print(f"\n交易统计:")
        print(f"   总交易: {total}")
        print(f"   盈利: {won} ({won/total*100:.1f}%)")
    
    return result


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../..')
    
    from ml_framework.data_loader import StockDataLoader
    from ml_framework.config import DATA_PATH
    import numpy as np
    
    loader = StockDataLoader(DATA_PATH)
    df = loader.load(years_back=1, select_codes=['000001', '000002', '600519'])
    df['pred_return'] = np.random.randn(len(df)) * 0.03
    
    result = run_threshold_backtest(df, buy_threshold=0.01, sell_threshold=-0.005)
