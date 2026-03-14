"""
ML Framework + Backtrader 集成回测
将 ml_framework 的预测结果传入 Backtrader 进行详细回测
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Signal:
    """交易信号"""
    date: pd.Timestamp
    code: str
    pred_return: float
    confidence: float = 1.0


class MLTopKStrategy(bt.Strategy):
    """
    Top-K 选股策略（基于ML预测）
    
    策略逻辑:
    1. 每日根据预测收益率选出 Top-K 股票
    2. 等权买入（或按预测收益加权）
    3. 持有到下一个调仓日
    """
    
    params = (
        ('signals', None),          # 信号字典 {date: [Signal, ...]}
        ('top_k', 10),              # 持仓数量
        ('rebalance_freq', 1),      # 调仓频率（1=每日，5=每周）
        ('commission', 0.001),      # 手续费
        ('print_log', True),
    )
    
    def __init__(self):
        self.signals = self.params.signals or {}
        self.current_holdings = set()
        self.rebalance_counter = 0
        
    def log(self, txt, dt=None):
        if self.params.print_log:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入 {order.data._name}: {order.executed.size}股 @ {order.executed.price:.2f}')
            else:
                self.log(f'卖出 {order.data._name}: {order.executed.size}股 @ {order.executed.price:.2f}')
    
    def next(self):
        """每日执行"""
        current_date = pd.Timestamp(self.datas[0].datetime.date(0))
        
        # 检查是否需要调仓
        self.rebalance_counter += 1
        if self.rebalance_counter % self.params.rebalance_freq != 0:
            return
        
        # 获取今日信号
        today_signals = self.signals.get(current_date, [])
        if not today_signals:
            return
        
        # 选出 Top-K
        sorted_signals = sorted(today_signals, key=lambda x: x.pred_return, reverse=True)
        top_k_codes = {s.code for s in sorted_signals[:self.params.top_k]}
        
        self.log(f'调仓 | 候选{len(today_signals)}只 | 选中{len(top_k_codes)}只')
        
        # 卖出不在 Top-K 中的持仓
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0 and data._name not in top_k_codes:
                self.close(data=data)
                self.log(f'  卖出 {data._name}（不在Top-{self.params.top_k}）')
        
        # 计算每只股票的仓位
        available_cash = self.broker.getcash()
        cash_per_stock = available_cash / max(len(top_k_codes), 1)
        
        # 买入新的 Top-K
        for signal in sorted_signals[:self.params.top_k]:
            code = signal.code
            for data in self.datas:
                if data._name == code:
                    pos = self.getposition(data)
                    if pos.size == 0:  # 还未持仓
                        size = int(cash_per_stock / data.close[0])
                        if size > 0:
                            self.buy(data=data, size=size)
                            self.log(f'  买入 {code}（预测收益:{signal.pred_return:.4f}）')
                    break


def prepare_signals(test_df: pd.DataFrame, 
                   pred_col: str = 'pred_return',
                   code_col: str = 'code',
                   date_col: str = 'date') -> Dict[pd.Timestamp, List[Signal]]:
    """
    将预测数据转换为信号字典
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        包含预测结果的测试集
    pred_col : str
        预测收益率列名
    code_col : str
        股票代码列名
    date_col : str
        日期列名
    
    Returns:
    --------
    signals : dict
        {date: [Signal, ...]}
    """
    signals = {}
    
    for date, group in test_df.groupby(date_col):
        daily_signals = []
        for _, row in group.iterrows():
            signal = Signal(
                date=date,
                code=str(row[code_col]),
                pred_return=row[pred_col],
                confidence=abs(row[pred_col])  # 用预测收益的绝对值作为置信度
            )
            daily_signals.append(signal)
        signals[date] = daily_signals
    
    return signals


def run_backtrader_with_predictions(test_df: pd.DataFrame,
                                    pred_col: str = 'pred_return',
                                    price_col: str = 'close',
                                    code_col: str = 'code',
                                    date_col: str = 'date',
                                    top_k: int = 10,
                                    initial_cash: float = 100000.0,
                                    commission: float = 0.001,
                                    print_log: bool = True) -> Dict:
    """
    使用 Backtrader 运行 ML 策略回测
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        测试集数据（包含真实价格和预测收益）
    pred_col : str
        预测收益率列名
    price_col : str
        价格列名（用于交易）
    code_col : str
        股票代码列名
    date_col : str
        日期列名
    top_k : int
        每日选股数量
    initial_cash : float
        初始资金
    commission : float
        手续费率
    print_log : bool
        是否打印日志
    
    Returns:
    --------
    result : dict
        回测结果字典
    """
    print("=" * 70)
    print("🚀 ML策略 Backtrader 回测")
    print("=" * 70)
    print(f"\n参数:")
    print(f"   初始资金: {initial_cash:,.2f}")
    print(f"   选股数量: Top-{top_k}")
    print(f"   手续费: {commission*100:.2f}%")
    
    # 准备信号
    signals = prepare_signals(test_df, pred_col, code_col, date_col)
    
    # 获取所有股票和日期
    all_codes = test_df[code_col].unique()
    all_dates = sorted(test_df[date_col].unique())
    
    print(f"\n数据概览:")
    print(f"   股票: {len(all_codes)} 只")
    print(f"   交易日: {len(all_dates)} 天 ({all_dates[0]} ~ {all_dates[-1]})")
    
    # 初始化 Cerebro
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(
        MLTopKStrategy,
        signals=signals,
        top_k=top_k,
        print_log=print_log
    )
    
    # 为每只股票添加数据
    for code in all_codes:
        stock_df = test_df[test_df[code_col] == code].copy()
        stock_df = stock_df.sort_values(date_col)
        stock_df.set_index(date_col, inplace=True)
        
        # 确保必要的列存在
        if price_col not in stock_df.columns:
            continue
            
        data = bt.feeds.PandasData(
            dataname=stock_df,
            datetime=None,
            open=price_col,
            high=price_col,
            low=price_col,
            close=price_col,
            volume=None,
            openinterest=-1
        )
        cerebro.adddata(data, name=str(code))
    
    # 设置资金和手续费
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 运行回测
    print("\n💰 开始回测...")
    print("-" * 70)
    
    results = cerebro.run()
    strat = results[0]
    
    print("-" * 70)
    final_value = cerebro.broker.getvalue()
    print(f"💰 最终资金: {final_value:,.2f}")
    
    # 获取分析结果
    returns_analyzer = strat.analyzers.returns.get_analysis()
    sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
    drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
    trades_analyzer = strat.analyzers.trades.get_analysis()
    
    result = {
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': (final_value / initial_cash - 1),
        'annual_return': returns_analyzer.get('rnorm', 0),
        'sharpe_ratio': sharpe_analyzer.get('sharperatio', 0) or 0,
        'max_drawdown': drawdown_analyzer['max']['drawdown'] if drawdown_analyzer['max'] else 0,
        'trades': trades_analyzer
    }
    
    # 打印详细结果
    print("\n" + "=" * 70)
    print("📊 回测结果")
    print("=" * 70)
    print(f"总收益率: {result['total_return']*100:.2f}%")
    print(f"年化收益率: {result['annual_return']*100:.2f}%")
    print(f"夏普比率: {result['sharpe_ratio']:.3f}")
    print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
    
    # 交易统计
    if trades_analyzer and trades_analyzer.get('total'):
        total_trades = trades_analyzer['total']['total']
        won_trades = trades_analyzer.get('won', {}).get('total', 0)
        lost_trades = trades_analyzer.get('lost', {}).get('total', 0)
        win_rate = won_trades / total_trades if total_trades > 0 else 0
        
        print(f"\n交易统计:")
        print(f"   总交易: {total_trades}")
        print(f"   盈利: {won_trades} ({win_rate*100:.1f}%)")
        print(f"   亏损: {lost_trades}")
    
    # 绘图（可选）
    # cerebro.plot(style='candlestick', barup='red', bardown='green')
    
    return result


def example_usage():
    """
    使用示例：结合 ml_framework 进行完整回测
    """
    from ml_framework.main import run_ml_pipeline
    from ml_framework.models import XGBoostModel
    
    print("=" * 70)
    print("示例: ML Pipeline + Backtrader 回测")
    print("=" * 70)
    
    # 1. 运行 ML Pipeline 获取预测结果
    print("\n1️⃣ 运行 ML Pipeline...")
    result = run_ml_pipeline(XGBoostModel, {'n_estimators': 50}, sample_n=20)
    
    # 获取 test_df 和预测结果（这里简化处理，实际需要修改 run_ml_pipeline 返回 test_df）
    # 假设 test_df 包含 'pred_return' 列
    
    # 2. 使用 Backtrader 回测
    print("\n2️⃣ 使用 Backtrader 回测...")
    # backtest_result = run_backtrader_with_predictions(test_df, top_k=5)
    
    print("\n✅ 示例完成")


if __name__ == '__main__':
    # 创建模拟数据进行演示
    print("创建模拟数据进行演示...")
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-01', freq='B')
    codes = ['000001', '000002', '600519', '002594', '300750']
    
    mock_data = []
    for date in dates:
        for code in codes:
            base_price = np.random.uniform(10, 100)
            pred_return = np.random.randn() * 0.02  # 模拟预测收益
            mock_data.append({
                'date': date,
                'code': code,
                'close': base_price,
                'pred_return': pred_return
            })
    
    test_df = pd.DataFrame(mock_data)
    
    # 运行回测
    result = run_backtrader_with_predictions(test_df, top_k=3, print_log=False)
    print("\n✅ 演示完成！")
