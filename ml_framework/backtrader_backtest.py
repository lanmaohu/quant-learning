"""
ML策略 Backtrader 回测系统
使用机器学习模型预测信号进行回测
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_framework.data_loader import StockDataLoader, time_series_split
from ml_framework.feature_engineering import FeatureEngineer
from ml_framework.config import DATA_PATH


class MLStrategy(bt.Strategy):
    """
    机器学习预测策略
    
    基于模型预测的未来收益率进行选股和择时
    """
    
    params = (
        ('pred_col', 'pred_return'),    # 预测收益率列名
        ('top_k', 5),                   # 每日持仓数量
        ('threshold', 0.001),           # 买入阈值（预测收益 > 0.1%）
        ('stop_loss', 0.05),            # 止损比例 5%
        ('take_profit', 0.10),          # 止盈比例 10%
        ('print_log', True),            # 是否打印日志
    )
    
    def __init__(self):
        self.order = None
        self.buy_prices = {}  # 记录每只股票的买入价
        
    def log(self, txt, dt=None):
        """日志输出"""
        if self.params.print_log:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """订单状态更新"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_prices[order.data._name] = order.executed.price
                self.log(f'【买入】{order.data._name} 价格:{order.executed.price:.2f} 数量:{order.executed.size}')
            else:
                buy_price = self.buy_prices.get(order.data._name, 0)
                if buy_price > 0:
                    pnl = (order.executed.price / buy_price - 1) * 100
                    self.log(f'【卖出】{order.data._name} 价格:{order.executed.price:.2f} 盈亏:{pnl:+.2f}%')
                del self.buy_prices[order.data._name]
        
        self.order = None
    
    def next(self):
        """每个交易日执行"""
        if self.order:
            return
        
        current_date = self.datas[0].datetime.date(0)
        
        # 收集所有股票的预测收益率
        predictions = []
        for data in self.datas:
            if len(data) > 0:
                # 假设预测值存储在 close 价格中（作为信号）
                # 实际使用时需要通过其他方式传递预测值
                pred_return = data.close[0]  # 这里需要调整
                predictions.append({
                    'code': data._name,
                    'pred_return': pred_return,
                    'close': data.close[0]
                })
        
        if not predictions:
            return
        
        # 按预测收益排序
        pred_df = pd.DataFrame(predictions).sort_values('pred_return', ascending=False)
        top_stocks = pred_df.head(self.params.top_k)['code'].tolist()
        
        # 获取当前持仓
        current_positions = {d._name: pos.size for d, pos in self.positions.items() if pos.size > 0}
        
        # 卖出不在 Top-K 中的股票
        for code in list(current_positions.keys()):
            if code not in top_stocks.values:
                for data in self.datas:
                    if data._name == code:
                        self.order = self.sell(data=data, size=current_positions[code])
                        break
        
        # 买入新的 Top-K 股票
        cash_per_stock = self.broker.getcash() / self.params.top_k
        for _, row in top_stocks.iterrows():
            code = row['code']
            if code not in current_positions and row['pred_return'] > self.params.threshold:
                for data in self.datas:
                    if data._name == code:
                        size = int(cash_per_stock / row['close'])
                        if size > 0:
                            self.order = self.buy(data=data, size=size)
                        break


class MLSignalData(bt.feeds.PandasData):
    """
    扩展的 PandasData，包含预测收益率字段
    """
    lines = ('pred_return',)  # 新增预测收益率线
    params = (
        ('pred_return', -1),  # -1 表示使用 close 列的索引
    )


def run_ml_backtest(pred_df: pd.DataFrame, 
                    initial_cash: float = 100000.0,
                    top_k: int = 5,
                    commission: float = 0.001,
                    print_log: bool = True) -> Dict:
    """
    运行 ML 策略回测
    
    Parameters:
    -----------
    pred_df : pd.DataFrame
        包含预测结果的数据框，列: [date, code, close, pred_return, ...]
    initial_cash : float
        初始资金
    top_k : int
        每日选股数量
    commission : float
        手续费率
    print_log : bool
        是否打印日志
    
    Returns:
    --------
    result : dict
        回测结果
    """
    print("=" * 70)
    print("🚀 ML策略回测 - Backtrader")
    print("=" * 70)
    print(f"\n参数配置:")
    print(f"   初始资金: {initial_cash:,.2f}")
    print(f"   每日选股: Top-{top_k}")
    print(f"   手续费: {commission*100:.2f}%")
    
    # 初始化 Cerebro
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(
        MLStrategy,
        top_k=top_k,
        print_log=print_log
    )
    
    # 为每只股票添加数据
    dates = pred_df['date'].unique()
    codes = pred_df['code'].unique()
    
    print(f"\n📊 数据概览:")
    print(f"   股票数量: {len(codes)}")
    print(f"   交易日: {len(dates)} ({dates[0]} ~ {dates[-1]})")
    
    for code in codes[:10]:  # 限制前10只避免太慢
        stock_df = pred_df[pred_df['code'] == code].copy()
        stock_df = stock_df.sort_values('date')
        stock_df.set_index('date', inplace=True)
        
        data = MLSignalData(
            dataname=stock_df,
            datetime=None,
            open='close',  # 使用收盘价作为开高低低（简化）
            high='close',
            low='close',
            close='close',
            volume='volume' if 'volume' in stock_df.columns else None,
            pred_return='pred_return',
            openinterest=-1
        )
        cerebro.adddata(data, name=code)
    
    # 设置初始资金
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 运行回测
    print("\n💰 开始回测...")
    print("-" * 70)
    
    results = cerebro.run()
    strat = results[0]
    
    print("-" * 70)
    print(f"💰 最终资金: {cerebro.broker.getvalue():,.2f}")
    
    # 获取分析结果
    returns = strat.analyzers.returns.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    
    result = {
        'initial_cash': initial_cash,
        'final_value': cerebro.broker.getvalue(),
        'total_return': returns['rtot'],
        'annual_return': returns['rnorm'],
        'sharpe_ratio': sharpe.get('sharperatio', 0),
        'max_drawdown': drawdown['max']['drawdown'],
        'trades': trades
    }
    
    # 打印结果
    print("\n" + "=" * 70)
    print("📊 回测结果")
    print("=" * 70)
    print(f"总收益率: {result['total_return']*100:.2f}%")
    print(f"年化收益率: {result['annual_return']*100:.2f}%")
    print(f"夏普比率: {result['sharpe_ratio']:.3f}")
    print(f"最大回撤: {result['max_drawdown']:.2f}%")
    
    if trades.get('total', {}).get('total', 0) > 0:
        print(f"\n交易统计:")
        print(f"   总交易: {trades['total']['total']}")
        print(f"   盈利: {trades.get('won', {}).get('total', 0)}")
        print(f"   亏损: {trades.get('lost', {}).get('total', 0)}")
    
    return result


def run_simple_backtest(model_name='xgboost', top_k=10):
    """
    简化版回测流程：训练模型 -> 预测 -> Backtrader回测
    """
    from ml_framework.main import run_ml_pipeline
    from ml_framework.models import MODEL_REGISTRY
    
    print("=" * 70)
    print(f"🚀 完整流程: {model_name} -> Backtrader回测")
    print("=" * 70)
    
    # 1. 运行 ML pipeline 获取预测结果
    ModelClass = MODEL_REGISTRY.get(model_name)
    if not ModelClass:
        raise ValueError(f"未知模型: {model_name}")
    
    # 这里简化处理，实际应该获取 test_features 的预测
    # 为了演示，使用模拟数据
    print("\n⚠️ 简化演示：使用模拟预测数据")
    print("   实际使用时，传入 ml_framework 的预测结果")
    
    # 创建模拟预测数据
    loader = StockDataLoader(DATA_PATH)
    df = loader.load(years_back=1)
    codes = loader.select_sample_codes(n=10)
    
    # 模拟预测（随机）
    np.random.seed(42)
    pred_data = []
    for date in df['date'].unique()[-30:]:  # 最近30天
        for code in codes[:5]:
            stock_data = df[(df['date'] == date) & (df['code'] == code)]
            if len(stock_data) > 0:
                pred_data.append({
                    'date': date,
                    'code': code,
                    'close': stock_data['close'].values[0],
                    'volume': stock_data['volume'].values[0],
                    'pred_return': np.random.randn() * 0.02  # 模拟预测收益
                })
    
    pred_df = pd.DataFrame(pred_data)
    
    # 2. 运行 Backtrader 回测
    result = run_ml_backtest(pred_df, top_k=top_k)
    
    return result


if __name__ == '__main__':
    # 示例运行
    result = run_simple_backtest(model_name='xgboost', top_k=5)
