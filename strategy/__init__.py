"""
策略模块包

包含基于 Backtrader 的量化交易策略实现。

主要入口函数：
    run_topk_backtest_ultra_optimized  — TopK 策略回测（推荐）
    run_threshold_backtest_ultra       — Threshold 策略回测（推荐）
    create_trade_chart                 — 策略净值图
    analyze_trade_profit_loss          — 交易盈亏分析
"""

from strategy.backtrader_topk_strategy import (
    run_topk_backtest_ultra_optimized,
    run_topk_backtest_optimized,
    run_topk_backtest,
    create_trade_chart,
)
from strategy.backtrader_threshold_strategy import (
    run_threshold_backtest_ultra,
    analyze_trade_profit_loss,
)

__all__ = [
    "run_topk_backtest_ultra_optimized",
    "run_topk_backtest_optimized",
    "run_topk_backtest",
    "run_threshold_backtest_ultra",
    "create_trade_chart",
    "analyze_trade_profit_loss",
]
